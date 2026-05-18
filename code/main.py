import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from RawDataAnalyzer import RawDataAnalyzer
from DatasetTransformer import DatasetTransformer
from OrdinalRegression import OrdinalRegression
from RegressorEvaluator import RegressorEvaluator
from LLMFineTuner import LLMFineTuner
from LLMResponseObtainer import LLMResponseObtainer
from LLMAggregator import LLMAggregator
from LLMEvaluator import LLMEvaluator
from PANDA import PANDA



#Fine-tuned
API_URL_MISTRAL_7B_4LAYERS = ""
API_URL_LLAMA3_2_3B_4LAYERS = ""
API_URL_QWEN2_5_4LAYERS = ""
API_URL_QWEN2_4LAYERS = ""

#Original
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.2"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
QWEN2_5_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN2_1_5B = "Qwen/Qwen2-1.5B-Instruct"

# Other models
other_model = ""

API_URL = other_model
HF_API_TOKEN = ""

paths_to_jsons = {
    "Fine-tuned models": {
        "Raw models": "./LLM evaluation/Fine-tuned models/Raw models",
        "Raw models + Rubric": "./LLM evaluation/Fine-tuned models/Raw models + Rubric",
        "Raw models + Rubric + PANDA": "./LLM evaluation/Fine-tuned models/Raw models + Rubric + PANDA",
        "Raw models + PANDA + Scores": "./LLM evaluation/Fine-tuned models/Raw models + PANDA + Scores"
    },
    "Non-fine-tuned models": {
        "Raw models": "./LLM evaluation/Non Fine-tuned models/Raw models",
        "Raw models + Rubric": "./LLM evaluation/Non Fine-tuned models/Raw models + Rubric",
        "Raw models + PANDA": "./LLM evaluation/Non Fine-tuned models/Raw models + PANDA",
        "Raw models + PANDA + Scores": "./LLM evaluation/Non Fine-tuned models/Raw models + PANDA + Scores",
        "Raw models + Rubric + PANDA": "./LLM evaluation/Non Fine-tuned models/Raw models + Rubric + PANDA",
        "Raw models + Rubric + PANDA + Scores": "./LLM evaluation/Non Fine-tuned models/Raw models + Rubric + PANDA + Scores"
    },
}

with open('config.json', 'r') as f:
    config = json.load(f)

switch = config["StageSwitch"]
def save_config():
    config["StageSwitch"] = switch
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

datasets_paths = config["DatasetsPaths"]
path_to_results = config["Results"]


if __name__ == "__main__":
    # Transforms the Hugging Face dataset by performing feature engineering and selection.
    data_transformer = DatasetTransformer(datasets_paths)
    if switch["transform_datasets"]:
        data_transformer.transform_hugging_face_dataset()
        switch["transform_datasets"] = 0
        save_config()
    transformed_hugging_face_dataset = data_transformer.load_transformed_hugging_face_dataset()



    # Analyzes the transformed dataset by printing its shape, checking for null values, and listing column names.
    dataset_analyzer = RawDataAnalyzer(dataset=transformed_hugging_face_dataset, 
                                    dataset_name="raw_hugging_face_dataset")
    criteria = ["task_achievement", 
            "coherence_and_cohesion", 
            "lexical_resource", 
            "grammatical_range_and_accuracy"]
    if switch["analyze_data"]:
        dataset_analyzer.print_dataset_shape()
        dataset_analyzer.print_rows_with_null()
        dataset_analyzer.print_column_names()    
        dataset_analyzer.plot_score_distributions(criteria=criteria, save_dir="./LLM evaluation")
        switch["analyze_data"] = 1
        save_config()



    # Prepares the data for training by splitting it into training and testing sets, 
    # and converting the essays to string format.
    x = transformed_hugging_face_dataset["essay"]
    y = transformed_hugging_face_dataset[["task_achievement", 
                              "coherence_and_cohesion", 
                              "lexical_resource", 
                              "grammatical_range_and_accuracy"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.1,
        random_state=0)
    x_train = x_train.astype(str).tolist()
    x_test = x_test.astype(str).tolist()

    # Criteria for training
    criteria = ["task_achievement", 
                "coherence_and_cohesion", 
                "lexical_resource", 
                "grammatical_range_and_accuracy"]



    if switch["train_custom_regressor"]:
        # # Prepare ordinal encoding
        # unique_bands = sorted(np.unique(np.concatenate([y_train, y_test])))
        # band2idx = {b:i for i,b in enumerate(unique_bands)}
        # idx2band = {i:b for b,i in band2idx.items()}
        # K = len(unique_bands)

        # # Create ordinal labels
        # y_train_ord = OrdinalRegression.ordinal_encode_multi(
        #     y_train, band2idx, K
        # )
        # y_test_ord = OrdinalRegression.ordinal_encode_multi(
        #     y_test, band2idx, K
        # )

        # # Create model
        # num_tasks = len(criteria)
        # ord_model = OrdinalRegression(
        #     num_classes=K,
        #     num_tasks=num_tasks
        # )

        # # Training
        # start_time = time.time()
        # ord_model.fit(x_train, y_train_ord, epochs=15, batch_size=16)
        # end_time = time.time()
        # print(f"Training time: {end_time - start_time:.2f} seconds")

        # # Prediction
        # y_pred = np.asarray(ord_model.predict(x_test, idx2band, batch_size=16))
        
        predictions_df  = pd.read_csv("C:/Users/serj/Documents/IU of Applied Sciences. Studying/Bachelor Thesis/Best regressor/roberta-base-results/results/predictions_table.csv")
        # extract columns
        y_true = predictions_df[[f"true_{c}" for c in criteria]].values
        y_pred = predictions_df[[f"pred_{c}" for c in criteria]].values
        # Evaluation
        path_to_ordinal_regression = path_to_results["path_to_ordinal_regression"]
        os.makedirs(path_to_ordinal_regression, exist_ok=True)
        unique_bands = np.unique(np.concatenate([y_train.values.flatten(), y_test.values.flatten()]))
        regressor_evaluator = RegressorEvaluator()
        regressor_evaluator.run_full_evaluation(
            y_true=y_test.values,
            y_pred=y_pred,
            criteria=criteria,
            bands=unique_bands,
            # loss_history=ord_model.loss_history,
            save_dir=path_to_ordinal_regression
        )
        # ord_model.save_model_and_tokenizer(ord_model, path_to_ordinal_regression)
        # switch["train_custom_regressor"] = 0
        save_config()



    if switch["fine_tune_llm"]:
        model_name = "mistralai/Mistral-7B-v0.1"
        trainer = LLMFineTuner(
            model_name=model_name,
            max_len=512,
            mode="lora"
        )

        trainer.train(
            x_train,
            y_train.to_numpy().tolist(),
        )
        trainer.evaluate_llm(
            x_test,
            y_test.to_numpy().tolist()
        )
        switch["fine_tine_llm"] = 0
        save_config()



    def convert_to_llm_dataset(texts: list[str], scores: list[list[float]], test_data: bool = False) -> pd.DataFrame:
        """
        Convert essay dataset into LLM SFT format.

        Parameters
        ----------
        texts : list[str]
            Essays.
        scores : list[list[float]]
            IELTS scores in order:
            [task_achievement, coherence_and_cohesion, lexical_resource, grammatical_range_and_accuracy]

        Returns
        -------
        pandas.DataFrame
            Dataset with columns:
            input_text, target_text
        """
        rows = []
        for essay, score in zip(texts, scores):
            target_json = {
                "task_achievement": score[0],
                "coherence_and_cohesion": score[1],
                "lexical_resource": score[2],
                "grammatical_range_and_accuracy": score[3],
            }

            if test_data:
                # For testing: separate column with score, leave empty string in prompt
                eval_column = json.dumps(target_json, ensure_ascii=False)
                answer_placeholder = ""  # <-- empty string after Answer
            else:
                # For training: putting JSON into prompt
                eval_column = None
                answer_placeholder = json.dumps(target_json, ensure_ascii=False)

            # Form prompt
            prompt = f"""
            ### Instruction:
            Evaluate the essay and return ONLY JSON, no text explanation.
            Scores range from 0 to 9 with step 0.5.

            Return format:
            {{
            "task_achievement": number,
            "coherence_and_cohesion": number,
            "lexical_resource": number,
            "grammatical_range_and_accuracy": number
            }}

            ### Text:
            {essay}

            ### Answer:
            {answer_placeholder}"""
            row = {
                "text": prompt.strip(),
                "evaluation": eval_column
            }
            rows.append(row)
        return pd.DataFrame(rows)



    if switch["get_llm_predictions"]:
        train_llm = convert_to_llm_dataset(texts = x_train, scores = y_train)
        test_llm = convert_to_llm_dataset(texts = x_test, scores = y_test, test_data=True)

        train_llm.to_csv("./datasets/train_llm.csv", index=False)
        test_llm.to_csv("./datasets/test_llm.csv", index=False)

        llm_response_obtainer = LLMResponseObtainer(SERVICE_TOKEN = HF_API_TOKEN)
        llm_response_obtainer.load_model_and_tokenizer(model_url=API_URL)
        llm_response_obtainer.get_responses_huggin_face(model_url = API_URL,
                                                        number_of_evaluations = 5,
                                                        test_dataset = x_test[:2])
        # llm_response_obtainer.get_responses_openai(model_url = API_URL,
        #                                         number_of_evaluations = 5,
        #                                         test_dataset = x_test[:2])
        switch["get_llm_predictions"] = 0
        save_config()
    


    # Evaluate LLM predictions using PANDA framework, which provides insights 
    # into the model's performance and identifies areas for improvement.
    if switch["PANDA_evaluation"]:
        panda_evaluator = PANDA(SERVICE_TOKEN=HF_API_TOKEN)
        panda_evaluator.load_model_and_tokenizer(model_url=API_URL)
        panda_evaluator.retrieve_insights(
            essays=np.array(x_train), 
            scores=np.array(y_train)
            )
        switch["PANDA_evaluation"] = 0
        save_config()
    

    # Evaluate the LLM's performance by comparing its predictions to the true scores,
    # and calculating metrics such as MAE, MSE, and QWK.
    if switch["aggregate_results"]:
        for key, path in paths_to_jsons["Fine-tuned models"].items():
            llm_aggregator = LLMAggregator(evaluation_folder_path = path)
            llm_aggregator.run(output_path = f"{path}/results.csv")
        y_test_df_true = y_test.reset_index(drop=True).copy()
        y_test_df_true.insert(0, "index", np.arange(len(y_test_df_true)))
        os.makedirs("./LLM evaluation/results",  exist_ok = True)
        y_test_df_true.to_csv("./LLM evaluation/results/y_test.csv", index=False)

    if switch["evaluate_llm"]:
        non_fine_tuned_models  = ["mistral_7B", "llama_32_3B", "qwen25", "qwen2"]
        non_fine_tuned_run_order = ["non_fine_tuned_raw_result", 
                                    "non_fine_tuned_raw_rubric_result", 
                                    "non_fine_tuned_raw_panda_result", 
                                    "non_fine_tuned_raw_rubric_panda_result",
                                    "non_fine_tuned_raw_panda_scores_result",
                                    "non_fine_tuned_raw_rubric_panda_scores_result"]
        fine_tuned_models = ["mistral_7B_4layers", "llama_32_3B_4layers", "qwen25_4layers", "qwen2_4layers"]
        fine_tuned_run_order = ["fine_tuned_raw_result", 
                                "fine_tuned_raw_rubric_result", 
                                "fine_tuned_raw_rubric_panda_result",
                                "fine_tuned_raw_panda_scores_result"]
        result_plots_path = "./result_plots"

        # Non-fine-tuned 
        llm_evaluator = LLMEvaluator(
            results_folder_path="./LLM evaluation/results/non_fine_tuned",
            y_true_path="./LLM evaluation/results/y_test.csv"
        )
        llm_evaluator.load_all(
            run_order = non_fine_tuned_run_order
        )
        llm_evaluator.plot_metrics(save_dir=f"{result_plots_path}/metrics/non_fine_tuned")

        # Fine-tuned
        llm_evaluator = LLMEvaluator(
            results_folder_path="./LLM evaluation/results/fine_tuned",
            y_true_path="./LLM evaluation/results/y_test.csv"
        )
        llm_evaluator.load_all(
            run_order = fine_tuned_run_order
        )
        llm_evaluator.plot_metrics(save_dir=f"{result_plots_path}/metrics/fine_tuned")

        save_config()

