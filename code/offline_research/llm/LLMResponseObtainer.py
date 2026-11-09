"""Collect essay scores from local models, Hugging Face endpoints and OpenAI."""

from typing import Any
import pandas as pd
import torch
import json
import os
from dataclasses import asdict

from ielts_judge.Paths import PATHS
from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from ielts_judge.Prompts import CONTEXT_SETUPS, JudgeContext, build_judge_messages, submission_parts
from ielts_judge.Scores import parse_scores
from ielts_judge.llm.Generation import generate_response
from offline_research.data.Runtime import Submission, SubmissionBatch, evaluation_samples, prediction_record



class LLMResponseObtainer:
    """A class to obtain responses from LLMs for essay evaluation."""
    def __init__(self, SERVICE_TOKEN: str, max_new_tokens: int = 256, *, setup: str="base", context_provider: Any=None) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            SERVICE_TOKEN (str): Service token used by this operation.
            max_new_tokens (int): Max new tokens used by this operation.
            setup (str): Context setup selecting rubric, PANDA references and AES guidance.
            context_provider (Any): Context provider used by this operation.

        Returns:
            None: This operation returns no value.
        """
        if setup not in CONTEXT_SETUPS:
            raise ValueError("Unknown context setup")
        self.setup, self.context_provider = setup, context_provider
        self.SERVICE_TOKEN = SERVICE_TOKEN
        self.max_new_tokens = max_new_tokens
        self.last_generation: dict = {}

    # Process locally
    def load_model_and_tokenizer(self, model_url: str) -> None:
        """Load the model and tokenizer for local evaluation.

        Args:
            model_url (str): Model url used by this operation.

        Returns:
            None: This operation returns no value.
        """
        from huggingface_hub import login
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.SERVICE_TOKEN:
            login(self.SERVICE_TOKEN)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_url, clean_up_tokenization_spaces=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_url,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    def _messages(self, submission: Any) -> Any:
        """Messages.

        Args:
            submission (Any): Submission used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        task, essay = submission_parts(submission)
        rubric, references, aes = CONTEXT_SETUPS[self.setup]
        supplied = self.context_provider(submission) if self.context_provider is not None else JudgeContext()
        if aes and not supplied.aes_priors:
            raise ValueError("AES setup needs real preliminary scores")
        context = JudgeContext(supplied.aes_priors if aes else {},
                               supplied.references if references else (), rubric)
        return build_judge_messages(task, essay, context)

    def __evaluate_essay_locally(self, essay: Submission) -> dict | None:
        """Evaluate essay locally.

        Args:
            essay (Submission): English essay text supplied for evaluation.

        Returns:
            dict | None: Named result fields and metadata produced by this operation.
        """
        self.last_generation = {}
        generation = generate_response(
            self.model, self.tokenizer, self._messages(essay),
            max_new_tokens=self.max_new_tokens, generation_settings={"do_sample": False},
            criterion_ids=PUBLIC_CRITERION_IDS,
        )
        self.last_generation = {**asdict(generation), "parse_error": None}
        try:
            return parse_scores(generation.raw_text)
        except ValueError:
            self.last_generation["parse_error"] = "invalid_integer_scores"
            return None

    def get_response_locally(self, number_of_evaluations: int,
                                   test_dataset: SubmissionBatch) -> None:
        """Get responses for a series of essays using the locally loaded model.

        Args:
            number_of_evaluations (int): Number of evaluations used by this operation.
            test_dataset (SubmissionBatch): Test dataset used by this operation.

        Returns:
            None: This operation returns no value.
        """
        output_dir = PATHS["results"]["llm_hugging_face_responses"]
        diagnostic_dir = os.path.join(
            output_dir, PATHS["files"]["llm_response_diagnostics"]
        )
        os.makedirs(diagnostic_dir, exist_ok=True)
        for evaluation in range(number_of_evaluations):
            results = []
            filename = PATHS["files"]["llm_response"].format(evaluation=evaluation + 1)
            diagnostic_path = os.path.join(diagnostic_dir, os.path.splitext(filename)[0] + ".jsonl")
            with open(diagnostic_path, "w", encoding="utf-8") as diagnostic_file:
                for sample_id, essay, manifest in evaluation_samples(test_dataset):
                    self.last_generation = {}
                    result = self.__evaluate_essay_locally(essay)
                    results.append(prediction_record(sample_id, essay, manifest, result))
                    record = {"sample_id": sample_id, "dataset_manifest": manifest, **self.last_generation, "prediction": result}
                    diagnostic_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    diagnostic_file.flush()
                    print(f"Inference {evaluation+1}: essay {sample_id}")
            with open(os.path.join(output_dir, filename),
                      "w", encoding="utf-8") as current_field:
                json.dump(results, current_field, ensure_ascii=False, indent=4)



    #Process online
    #Hugging Face
    def __evaluate_essay_hugging_face(self, model_url: str, essay: Submission) -> dict|None:
        """Evaluate an essay using the Hugging Face API.

        Args:
            model_url (str): Model url used by this operation.
            essay (Submission): English essay text supplied for evaluation.

        Returns:
            dict | None: Named result fields and metadata produced by this operation.
        """
        headers = {"Authorization": f"Bearer {self.SERVICE_TOKEN}"}
        messages = self._messages(essay)
        payload = {
            "inputs": messages,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "return_full_text": False
            }
        }

        import requests

        response = requests.post(
            model_url, headers=headers, json=payload, timeout=60
        )
        response.raise_for_status()
        output = response.json()

        # 👇 зависит от типа endpoint!
        if isinstance(output, list):
            text = output[0]["generated_text"]
        elif isinstance(output, dict) and "generated_text" in output:
            text = output["generated_text"]
        else:
            print("Неожиданный формат:", output)
            return None

        try:
            return parse_scores(text)
        except ValueError:
            return None

    def get_responses_huggin_face(self, model_url: str, number_of_evaluations: int,
                                  test_dataset: SubmissionBatch) -> None:
        """Get responses for a series of essays using the Hugging Face API.

        Args:
            model_url (str): Model url used by this operation.
            number_of_evaluations (int): Number of evaluations used by this operation.
            test_dataset (SubmissionBatch): Test dataset used by this operation.

        Returns:
            None: This operation returns no value.
        """
        for evaluation in range(number_of_evaluations):
            results = []
            for sample_id, essay, manifest in evaluation_samples(test_dataset):
                result = self.__evaluate_essay_hugging_face(model_url=model_url, essay=essay)
                results.append(prediction_record(sample_id, essay, manifest, result))
                print(f"Inference {evaluation+1}: essay {sample_id}")
            output_dir = PATHS["results"]["llm_hugging_face_responses"]
            os.makedirs(output_dir, exist_ok=True)
            filename = PATHS["files"]["llm_response"].format(evaluation=evaluation + 1)
            with open(os.path.join(output_dir, filename),
                      "w", encoding="utf-8") as current_field:
                json.dump(results, current_field, ensure_ascii=False, indent=4)

    #OpenAi
    def __connect_to_openai(self, model_url: str) -> None:
        """Connect to the OpenAI API using the provided model URL.

        Args:
            model_url (str): Model url used by this operation.

        Returns:
            None: This operation returns no value.
        """
        from openai import OpenAI

        self.client = OpenAI(
            base_url=model_url,
            api_key=self.SERVICE_TOKEN
        )

    def __evaluate_essay_openai(self, essay: Submission) -> Any:
        """Evaluate an essay using the OpenAI API.

        Args:
            essay (Submission): English essay text supplied for evaluation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from openai.types.chat import ChatCompletionMessageParam
        from typing import cast
        messages = cast(list[ChatCompletionMessageParam], self._messages(essay))

        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=0.0
            )
        except Exception as current_error:
            print("Ошибка запроса:", current_error)
            return None

        text = response.choices[0].message.content
        if text is None:
            return None
        try:
            return parse_scores(text)
        except ValueError:
            return None

    def get_responses_openai(self, model_url: str, number_of_evaluations: int,
                                  test_dataset: SubmissionBatch) -> None:
        """Get responses for a series of essays using the OpenAI API.

        Args:
            model_url (str): Model url used by this operation.
            number_of_evaluations (int): Number of evaluations used by this operation.
            test_dataset (SubmissionBatch): Test dataset used by this operation.

        Returns:
            None: This operation returns no value.
        """
        self.__connect_to_openai(model_url=model_url)
        for evaluation in range(number_of_evaluations):
            results = []
            for sample_id, essay, manifest in evaluation_samples(test_dataset):
                result = self.__evaluate_essay_openai(essay=essay)
                results.append(prediction_record(sample_id, essay, manifest, result))
                print(f"Inference {evaluation+1}: essay {sample_id}")
            output_dir = PATHS["results"]["llm_openai_responses"]
            os.makedirs(output_dir, exist_ok=True)
            filename = PATHS["files"]["llm_response"].format(evaluation=evaluation + 1)
            with open(os.path.join(output_dir, filename),
                      "w", encoding="utf-8") as current_field:
                json.dump(results, current_field, ensure_ascii=False, indent=4)

