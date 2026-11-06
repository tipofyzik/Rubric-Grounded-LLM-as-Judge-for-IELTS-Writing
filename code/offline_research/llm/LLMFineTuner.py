from typing import Any
import torch
import json
import re
import os
from typing import Sequence

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from ielts_judge.Criteria import CRITERION_IDS, PUBLIC_CRITERION_IDS, format_score_schema
from ielts_judge.Prompts import JudgeContext, build_judge_messages, submission_parts
from ielts_judge.Scores import parse_scores
from ielts_judge.llm.Generation import generate_response
from offline_research.data.Runtime import Submission, submission_text
from ielts_judge.Paths import PATHS



class LLMFineTuner:
    """Universal trainer for fine-tuning language models using LoRA or QLoRA.

    Attributes:
        max_len (int): Maximum sequence length for tokenized inputs.
        mode (str): Fine-tuning mode, either "lora" or "qlora".
        tokenizer (AutoTokenizer): HuggingFace tokenizer for the model.
        model (AutoModelForCausalLM): Language model to be fine-tuned.
    """

    def __init__(self, model_name: str, max_len: int = 1536,
                 mode: str = "lora") -> None:   # "lora" | "qlora"
        """Initialize this component's settings and collaborators.

        Args:
            model_name (str): Model name used by this operation.
            max_len (int): Max len used by this operation.
            mode (str): Mode used by this operation.

        Returns:
            None: This operation returns no value.
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
        """Apply LoRA configuration to the model for parameter-efficient fine-tuning.

        Returns:
            None: This operation returns no value.
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

    def _build_prompt(self, text: Submission, scores: list[float]) -> str:
        """Construct a structured prompt for the model based on input text and scores.

        Args:
            text (Submission): Text used by this operation.
            scores (list[float]): Four criterion scores using the public IELTS score contract.

        Returns:
            str: The computed result for the supplied inputs.
        """
        text = submission_text(text)
        output = dict(zip(CRITERION_IDS, scores))

        prompt = f"""
        ### Instruction:
        Evaluate the essay and return ONLY JSON, no text explanation".

        Return format:
        {format_score_schema()}

        ### Text:
        {text}

        ### Answer:
        """

        return prompt + json.dumps(output)

    # Dataset

    def build_dataset(self, essays: Sequence[Submission], scores: list[list[float]]) -> Dataset:
        """Build a HuggingFace Dataset from lists of texts and their corresponding scores.

        Args:
            essays (Sequence[Submission]): Essays used by this operation.
            scores (list[list[float]]): Four criterion scores using the public IELTS score contract.

        Returns:
            Dataset: The computed result for the supplied inputs.
        """
        prompts = [
            self._build_prompt(essay, score)
            for essay, score in zip(essays, scores)
            ]

        dataset = Dataset.from_dict({"text": prompts})

        def tokenize(example: dict) -> dict[str, Any]:
            """Tokenizes a text example for a text generation task, splitting it into prompt and answer parts.

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

            Args:
                example (dict): Example used by this operation.

            Returns:
                dict[str, Any]: Named result fields and metadata produced by this operation.
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

    def train(self, x_train: Sequence[Submission], y_train: list[list[float]]) -> None:
        """Fine-tune the model on the provided training data.

        Args:
            x_train (Sequence[Submission]): X train used by this operation.
            y_train (list[list[float]]): Y train used by this operation.

        Returns:
            None: This operation returns no value.
        """
        train_dataset = self.build_dataset(x_train, y_train)

        current_arguments = TrainingArguments(
            output_dir=PATHS["results"]["llm_training"],
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
            args=current_arguments,
            train_dataset=train_dataset
        )

        trainer.train()

    # Evaluation

    def evaluate_llm(self, x_test: Sequence[Submission], y_test: list[list[float]],
                     output_file: str = PATHS["results"]["llm_predictions"]) -> list[dict]:
        """Evaluate the fine-tuned model on test data and save results to a JSON file.

        Args:
            x_test (Sequence[Submission]): X test used by this operation.
            y_test (list[list[float]]): Y test used by this operation.
            output_file (str): Output file used by this operation.

        Returns:
            list[dict]: The computed result for the supplied inputs.
        """
        self.model.eval()
        results = []

        for text, true_scores in zip(x_test, y_test):

            submission = text
            task, essay = submission_parts(submission)
            messages = build_judge_messages(task, essay, JudgeContext(include_rubric=False))
            generation = generate_response(
                self.model, self.tokenizer, messages, max_new_tokens=256,
                generation_settings={"do_sample": False}, criterion_ids=PUBLIC_CRITERION_IDS,
            )
            decoded = generation.raw_text
            try:
                current_prediction = parse_scores(decoded)
                answer_part = json.dumps(current_prediction)
            except ValueError:
                current_prediction, answer_part = None, ""
            text = submission_text(submission)

            results.append({
                "essay_id": submission.get('essay_id') if isinstance(submission, dict) else None,
                "dataset_manifest": submission.get('dataset_manifest') if isinstance(submission, dict) else None,
                "text": text,
                "true_scores": true_scores,
                "full_model_output": decoded,
                "answer_extracted": answer_part,
                "prediction_json": current_prediction
            })
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as current_field:
            json.dump(results, current_field, ensure_ascii=False, indent=2)
        print(f"Saved results to {output_file}")

        return results
