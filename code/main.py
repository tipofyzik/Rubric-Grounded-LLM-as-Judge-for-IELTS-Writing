import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
from ielts_judge.paths import PATHS

# Fine-tuned
API_URL_MISTRAL_7B_4LAYERS = ""
API_URL_LLAMA3_2_3B_4LAYERS = ""
API_URL_QWEN2_5_4LAYERS = ""
API_URL_QWEN2_4LAYERS = ""

# Original
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.2"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
QWEN2_5_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN2_1_5B = "Qwen/Qwen2-1.5B-Instruct"

# Other models
other_model = ""

API_URL = other_model
HF_API_TOKEN = ""

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

switch = config["stage_switch"]


def save_config():
    config["stage_switch"] = switch
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


datasets_paths = {
    "hugging_face_train_dataset_path": PATHS["datasets"]["hugging_face_train"],
    "hugging_face_test_dataset_path": PATHS["datasets"]["hugging_face_test"],
    "kaggle_dataset_path": PATHS["datasets"]["kaggle"],
    "transformed_hugging_face_dataset_path": PATHS["datasets"][
        "transformed_hugging_face"
    ],
    "transformed_kaggle_dataset_path": PATHS["datasets"]["transformed_kaggle"],
}
path_to_results = {"path_to_ordinal_regression": PATHS["results"]["ordinal_regression"]}


if __name__ == "__main__":
    from ielts_judge.criteria import CRITERION_IDS, format_score_schema
    from ielts_judge.data.DatasetTransformer import DatasetTransformer

    # Transforms the Hugging Face dataset by performing feature engineering and selection.
    data_transformer = DatasetTransformer(datasets_paths)
    if switch["transform_datasets"]:
        data_transformer.transform_hugging_face_dataset()
        switch["transform_datasets"] = 0
        save_config()
    transformed_hugging_face_dataset = (
        data_transformer.load_transformed_hugging_face_dataset()
    )

    # Analyzes the transformed dataset by printing its shape, checking for null values, and listing column names.
    criteria = list(CRITERION_IDS)
    if switch["analyze_data"]:
        from ielts_judge.data.RawDataAnalyzer import RawDataAnalyzer

        dataset_analyzer = RawDataAnalyzer(
            dataset=transformed_hugging_face_dataset,
            dataset_name="raw_hugging_face_dataset",
        )
        dataset_analyzer.print_dataset_shape()
        dataset_analyzer.print_rows_with_null()
        dataset_analyzer.print_column_names()
        dataset_analyzer.plot_score_distributions(
            criteria=criteria, save_dir=PATHS["results"]["llm_evaluation"]
        )
        switch["analyze_data"] = 1
        save_config()

    # Prepares the data for training by splitting it into training and testing sets,
    # and converting the essays to string format.
    x = transformed_hugging_face_dataset["essay"]
    y = transformed_hugging_face_dataset[list(CRITERION_IDS)]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=0
    )
    x_train = x_train.astype(str).tolist()
    x_test = x_test.astype(str).tolist()

    # Criteria for training
    criteria = list(CRITERION_IDS)

    aes_scores = None
    if switch["train_custom_regressor"]:
        from ielts_judge.aes.RegressorEvaluator import RegressorEvaluator

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

        predictions_df = pd.read_csv(PATHS["external"]["aes_predictions"])
        # extract columns
        y_true = predictions_df[[f"true_{c}" for c in criteria]].values
        y_pred = predictions_df[[f"pred_{c}" for c in criteria]].values
        # Evaluation
        path_to_ordinal_regression = path_to_results["path_to_ordinal_regression"]
        os.makedirs(path_to_ordinal_regression, exist_ok=True)
        unique_bands = np.unique(
            np.concatenate([y_train.values.flatten(), y_test.values.flatten()])
        )
        regressor_evaluator = RegressorEvaluator()
        # regressor_evaluator.run_full_evaluation(
        #     y_true=y_test.values,
        #     y_pred=y_pred,
        #     criteria=criteria,
        #     bands=unique_bands,
        #     # loss_history=ord_model.loss_history,
        #     save_dir=path_to_ordinal_regression,
        # )
        aes_scores = regressor_evaluator.get_metrics_per_criterion(
            y_true, y_pred, criteria
        )
        # ord_model.save_model_and_tokenizer(ord_model, path_to_ordinal_regression)
        # switch["train_custom_regressor"] = 0
        save_config()

    if switch["fine_tune_llm"]:
        from ielts_judge.llm.LLMFineTuner import LLMFineTuner

        model_name = "mistralai/Mistral-7B-v0.1"
        trainer = LLMFineTuner(model_name=model_name, max_len=512, mode="lora")

        trainer.train(
            x_train,
            y_train.to_numpy().tolist(),
        )
        trainer.evaluate_llm(x_test, y_test.to_numpy().tolist())
        switch["fine_tune_llm"] = 0
        save_config()

    def convert_to_llm_dataset(
        texts: list[str], scores: list[list[float]], test_data: bool = False
    ) -> pd.DataFrame:
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
            target_json = dict(zip(CRITERION_IDS, score))

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
            {format_score_schema()}

            ### Text:
            {essay}

            ### Answer:
            {answer_placeholder}"""
            row = {"text": prompt.strip(), "evaluation": eval_column}
            rows.append(row)
        return pd.DataFrame(rows)

    if switch["get_llm_predictions"]:
        from ielts_judge.llm.LLMResponseObtainer import LLMResponseObtainer

        train_llm = convert_to_llm_dataset(
            texts=x_train, scores=y_train.to_numpy().tolist()
        )
        test_llm = convert_to_llm_dataset(
            texts=x_test, scores=y_test.to_numpy().tolist(), test_data=True
        )

        train_llm.to_csv(PATHS["datasets"]["llm_train"], index=False)
        test_llm.to_csv(PATHS["datasets"]["llm_test"], index=False)

        llm_response_obtainer = LLMResponseObtainer(SERVICE_TOKEN=HF_API_TOKEN)
        llm_response_obtainer.load_model_and_tokenizer(model_url=API_URL)
        llm_response_obtainer.get_responses_huggin_face(
            model_url=API_URL, number_of_evaluations=5, test_dataset=x_test[:2]
        )
        # llm_response_obtainer.get_responses_openai(model_url = API_URL,
        #                                         number_of_evaluations = 5,
        #                                         test_dataset = x_test[:2])
        switch["get_llm_predictions"] = 0
        save_config()

    # Evaluate LLM predictions using PANDA framework, which provides insights
    # into the model's performance and identifies areas for improvement.
    if switch["panda_evaluation"]:
        from ielts_judge.panda.Panda import PANDA

        panda_evaluator = PANDA(service_token=HF_API_TOKEN)
        panda_evaluator.load_model_and_tokenizer(model_url=API_URL)
        panda_evaluator.retrieve_insights(
            essays=np.array(x_train), scores=np.array(y_train)
        )
        switch["panda_evaluation"] = 0
        save_config()

    # Evaluate the LLM's performance by comparing its predictions to the true scores,
    # and calculating metrics such as MAE, MSE, and QWK.
    if switch["aggregate_results"]:
        from ielts_judge.evaluation.LLMAggregator import LLMAggregator

        for key, path in PATHS["evaluation_sources"]["fine_tuned"].items():
            llm_aggregator = LLMAggregator(evaluation_folder_path=path)
            llm_aggregator.run(
                output_path=os.path.join(path, PATHS["files"]["llm_aggregate"])
            )
        y_test_df_true = y_test.reset_index(drop=True).copy()
        y_test_df_true.insert(0, "index", np.arange(len(y_test_df_true)))
        llm_evaluation_results = PATHS["results"]["llm_evaluation_results"]
        os.makedirs(llm_evaluation_results, exist_ok=True)
        y_test_df_true.to_csv(
            os.path.join(
                llm_evaluation_results, PATHS["files"]["llm_evaluation_truth"]
            ),
            index=False,
        )

    if switch["evaluate_llm"]:
        from ielts_judge.evaluation.LLMEvaluator import LLMEvaluator

        non_fine_tuned_models = ["mistral_7B", "llama_32_3B", "qwen25", "qwen2"]
        non_fine_tuned_run_order = [
            "non_fine_tuned_raw_result",
            "non_fine_tuned_raw_rubric_result",
            "non_fine_tuned_raw_panda_result",
            "non_fine_tuned_raw_rubric_panda_result",
            "non_fine_tuned_raw_panda_scores_result",
            "non_fine_tuned_raw_rubric_panda_scores_result",
        ]
        fine_tuned_models = [
            "mistral_7B_4layers",
            "llama_32_3B_4layers",
            "qwen25_4layers",
            "qwen2_4layers",
        ]
        fine_tuned_run_order = [
            "fine_tuned_raw_result",
            "fine_tuned_raw_rubric_result",
            "fine_tuned_raw_rubric_panda_result",
            "fine_tuned_raw_panda_scores_result",
        ]
        llm_evaluation_results = PATHS["results"]["llm_evaluation_results"]
        y_test_path = os.path.join(
            llm_evaluation_results, PATHS["files"]["llm_evaluation_truth"]
        )

        # Non-fine-tuned
        llm_evaluator = LLMEvaluator(
            results_folder_path=PATHS["results"]["llm_non_fine_tuned_results"],
            y_true_path=y_test_path,
            aes_expert_metrics=aes_scores,
        )
        llm_evaluator.load_all(run_order=non_fine_tuned_run_order)
        llm_evaluator.plot_metrics(save_dir=PATHS["results"]["non_fine_tuned_plots"])

        # Fine-tuned
        llm_evaluator = LLMEvaluator(
            results_folder_path=PATHS["results"]["llm_fine_tuned_results"],
            y_true_path=y_test_path,
            aes_expert_metrics=aes_scores,
        )
        llm_evaluator.load_all(run_order=fine_tuned_run_order)
        llm_evaluator.plot_metrics(save_dir=PATHS["results"]["fine_tuned_plots"])

        save_config()
