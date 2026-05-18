import torch
import json
import re

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)



class LLMFineTuner:
    """
    Universal trainer for fine-tuning language models using LoRA or QLoRA.

    Attributes:
        max_len (int): Maximum sequence length for tokenized inputs.
        mode (str): Fine-tuning mode, either "lora" or "qlora".
        tokenizer (AutoTokenizer): HuggingFace tokenizer for the model.
        model (AutoModelForCausalLM): Language model to be fine-tuned.
    """

    def __init__(self, model_name: str, max_len: int = 1536,
                 mode: str = "lora") -> None:   # "lora" | "qlora"
        """
        Initialize the fine-tuner, load the model and tokenizer, and apply LoRA if needed.

        Args:
            model_name (str): Name or path of the HuggingFace pretrained model.
            max_len (int, optional): Maximum token sequence length. Defaults to 1536.
            mode (str, optional): Fine-tuning mode: "lora" or "qlora". Defaults to "lora".
        """
        self.max_len = max_len
        self.mode = mode

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if mode == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        self.apply_lora()

    def apply_lora(self) -> None:
        """
        Apply LoRA configuration to the model for parameter-efficient fine-tuning.
        """
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"
            ]
        )
            # q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head
        self.model = get_peft_model(self.model, config)   # type: ignore

    # Prompt building

    def _build_prompt(self, text: str, scores: list[float]) -> str:
        """
        Construct a structured prompt for the model based on input text and scores.

        Args:
            text (str): The input text to evaluate.
            scores (list[float]): A list of four evaluation scores in the order:
                [task_achievement, coherence_and_cohesion, lexical_resource, grammatical_range_and_accuracy].

        Returns:
            str: A formatted prompt including the JSON output of scores.
        """
        output = {
            "task_achievement": scores[0],
            "coherence_and_cohesion": scores[1],
            "lexical_resource": scores[2],
            "grammatical_range_and_accuracy": scores[3]
        }

        prompt = f"""
        ### Instruction:
        Evaluate the essay and return ONLY JSON, no text explanation".

        Return format:
        {{
        "task_achievement": number,
        "coherence_and_cohesion": number,
        "lexical_resource": number,
        "grammatical_range_and_accuracy": number
        }}

        ### Text:
        {text}

        ### Answer:
        """

        return prompt + json.dumps(output)

    # Dataset

    def build_dataset(self, essays: list[str], scores: list[list[float]]) -> Dataset:
        """
        Build a HuggingFace Dataset from lists of texts and their corresponding scores.

        Args:
            essays (list[str]): List of input texts.
            scores (list[list[float]]): List of score lists corresponding to each text.
            y (list[list[float]]): List of score lists corresponding to each text.

        Returns:
            Dataset: Tokenized dataset ready for model training.
        """
        prompts = [
            self._build_prompt(essay, score)
            for essay, score in zip(essays, scores)
            ]

        dataset = Dataset.from_dict({"text": prompts})

        def tokenize(example: dict):
            """
            Tokenizes a text example for a text generation task, splitting it into prompt and answer parts.

            This function expects the input `example` to be a dictionary with a key "text",
            containing the full text in the format:
                <prompt> ### Answer: <answer>

            Steps performed:

            1. Split the text into `prompt_part` and `answer_part` using the marker "### Answer:":
            - `prompt_part` includes everything up to and including "### Answer:".
            - `answer_part` includes everything after "### Answer:".
            
            2. Tokenize both parts using `self.tokenizer`:
            - `prompt_tokens` for the prompt part.
            - `answer_tokens` for the answer part.

            3. Construct the training inputs:
            - `input_ids`: concatenation of prompt tokens and answer tokens.
            - `labels`: assign -100 to prompt tokens (ignored during loss calculation)
                        and use the actual token IDs for answer tokens.
            - `attention_mask`: 1 for all actual tokens to indicate they should be attended to.

            4. Apply padding or truncation to match `self.max_len`:
            - If the sequence is shorter than `self.max_len`, pad `input_ids` with the tokenizer's
                pad token, `labels` with -100, and `attention_mask` with 0.
            - If the sequence is longer than `self.max_len`, truncate all lists to `self.max_len`.

            Returns:
                dict: A dictionary containing:
                    - "input_ids" (List[int]): Token IDs for both prompt and answer, padded/truncated.
                    - "labels" (List[int]): Labels for computing loss (-100 for prompt tokens).
                    - "attention_mask" (List[int]): Attention mask (1 for actual tokens, 0 for padding).
            """
            full = example["text"]
            answer_start = full.find("### Answer:")

            prompt_part = full[:answer_start + len("### Answer:")]
            answer_part = full[answer_start + len("### Answer:"):]

            prompt_tokens = self.tokenizer(prompt_part)
            answer_tokens = self.tokenizer(answer_part)

            input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
            labels = [-100] * len(prompt_tokens["input_ids"]) + answer_tokens["input_ids"]
            attention_mask = [1] * len(input_ids)

            pad_len = self.max_len - len(input_ids)
            if pad_len > 0:
                input_ids += [self.tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len
                attention_mask += [0] * pad_len
            else:
                input_ids = input_ids[:self.max_len]
                labels = labels[:self.max_len]
                attention_mask = attention_mask[:self.max_len]

            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            }

        return dataset.map(tokenize, remove_columns=["text"])

    # Training

    def train(self, x_train: list[str], y_train: list[list[float]]) -> None:
        """
        Fine-tune the model on the provided training data.

        Args:
            x_train (list[str]): List of training texts.
            y_train (list[list[float]]): Corresponding list of scores for training texts.
        """
        train_dataset = self.build_dataset(x_train, y_train)

        args = TrainingArguments(
            output_dir="./results/llm",
            learning_rate=2e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            logging_steps=20,
            save_strategy="epoch",
            fp16=True
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset
        )

        trainer.train()

    # Evaluation

    def evaluate_llm(self, x_test: list[str], y_test: list[list[float]],
                     output_file: str = "./results/llm_predictions.json") -> list[dict]:
        """
        Evaluate the fine-tuned model on test data and save results to a JSON file.

        Args:
            x_test (list[str]): List of test texts.
            y_test (list[list[float]]): Corresponding list of ground-truth scores.
            output_file (str, optional): Path to save evaluation results as JSON. Defaults to "./results/llm_predictions.json".

        Returns:
            list[dict]: A list of dictionaries containing:
                - "text": original input text,
                - "true_scores": ground-truth scores,
                - "full_model_output": full raw model output,
                - "answer_extracted": extracted JSON answer from the model,
                - "prediction_json": parsed JSON prediction (or None if parsing failed).
        """
        self.model.eval()
        results = []

        for text, true_scores in zip(x_test, y_test):

            prompt = f"""
            ### Instruction:
            Evaluate the essay and return ONLY JSON, no text explanation".

            Return format:
            {{
            "task_achievement": number,
            "coherence_and_cohesion": number,
            "lexical_resource": number,
            "grammatical_range_and_accuracy": number
            }}

            ### Text:
            {text}

            ### Answer:
            """

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_len
            ).to(self.model.device)

            with torch.no_grad():

                output = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            decoded = self.tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )

            match = re.search(r"\{.*\}", decoded, flags=re.DOTALL)
            if match:
                answer_part = match.group(0)
                try:
                    pred = json.loads(answer_part)
                except json.JSONDecodeError:
                    pred = None
            else:
                answer_part = ""
                pred = None

            results.append({
                "text": text,
                "true_scores": true_scores,
                "full_model_output": decoded,
                "answer_extracted": answer_part,
                "prediction_json": pred
            })
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {output_file}")

        return results