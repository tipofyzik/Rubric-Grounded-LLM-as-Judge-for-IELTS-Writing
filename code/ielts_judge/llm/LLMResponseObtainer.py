import pandas as pd
import torch
import json
import re
import os

from ..paths import PATHS
from ..criteria import format_score_schema



class LLMResponseObtainer:
    """
    A class to obtain responses from LLMs for essay evaluation.
    """
    def __init__(self, SERVICE_TOKEN: str, max_new_tokens: int = 200) -> None:
        """
        Initialize the LLMResponseObtainer with the necessary service token.

        Args:
            SERVICE_TOKEN (str): The API token for accessing the LLM service.
            max_new_tokens (int): Maximum number of generated tokens per local answer.

        Returns:
            None
        """
        self.SERVICE_TOKEN = SERVICE_TOKEN
        self.max_new_tokens = max_new_tokens
        self.last_generation: dict = {}

    # Process locally
    def load_model_and_tokenizer(self, model_url: str) -> None:
        """
        Load the model and tokenizer for local evaluation.

        Args:
            model_url (str): The URL or identifier of the model to load.

        Returns:
            None
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

    def __evaluate_essay_locally(self, essay: str) -> dict | None:
        """
        Evaluate an essay locally using the loaded model.

        Args:
            essay (str): The essay to evaluate.

        Returns:
            dict|None: The evaluation results or None if an error occurs.
        """
        from .JsonEosBlocker import JsonEosBlocker

        self.last_generation = {}
        prompt = f"""
        ### Instruction:
        Evaluate the essay and return ONLY JSON, no text explanation.
        Scores range from 0 to 9 with step 0.5.
        
        Return format:
        {format_score_schema()}
        
        ### Text:
        {essay}
        
        ### Answer:
        """
        messages = [
            {"role": "system", "content": "You are an IELTS examiner. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        generation_config = getattr(self.model, "generation_config", None)
        eos_tokens = getattr(generation_config, "eos_token_id", None)
        if eos_tokens is None:
            eos_tokens = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_tokens, int):
            eos_tokens = [eos_tokens]
        eos_blocker = JsonEosBlocker(self.tokenizer, input_length, eos_tokens or [])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.2,
                do_sample=True,
                logits_processor=[eos_blocker],
            )
        generated_tokens = outputs[0, input_length:]
        response = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        token_count = int(generated_tokens.numel())
        last_token = int(generated_tokens[-1].item()) if token_count else None
        if last_token is not None and last_token in (eos_tokens or []):
            stop_reason = "eos"
        elif token_count >= self.max_new_tokens:
            stop_reason = "max_new_tokens"
        else:
            stop_reason = "other"
        self.last_generation = {
            "raw_text": response,
            "input_tokens": int(input_length),
            "generated_tokens": token_count,
            "max_new_tokens": self.max_new_tokens,
            "last_token_id": last_token,
            "stop_reason": stop_reason,
            "parse_error": None,
        }
        try:
            object_start = response.find("{")
            if object_start < 0:
                raise ValueError("JSON object was not found in the generated answer.")
            answer, _ = json.JSONDecoder().raw_decode(response[object_start:])
            if not isinstance(answer, dict):
                raise ValueError("Generated JSON must be an object.")
            return answer
        except ValueError as error:
            self.last_generation["parse_error"] = str(error)
            print(f"JSON incomplete or invalid; tokens={token_count}/{self.max_new_tokens}; "
                  f"stop={stop_reason}: {response}")
            return None

    def get_response_locally(self, number_of_evaluations: int, 
                                   test_dataset: pd.Series) -> None:
        """
        Get responses for a series of essays using the locally loaded model.

        Args:
            number_of_evaluations (int): Number of repeated inference passes.
            test_dataset (pd.Series): The series of essays to evaluate.

        Returns:
            None
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
                for i, essay in enumerate(test_dataset):
                    self.last_generation = {}
                    result = self.__evaluate_essay_locally(essay)
                    results.append(result)
                    record = {"sample_id": i, **self.last_generation, "prediction": result}
                    diagnostic_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    diagnostic_file.flush()
                    print(f"Inference {evaluation+1}: essay {i+1}")
            with open(os.path.join(output_dir, filename),
                      "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)



    #Process online
    #Hugging Face
    def __evaluate_essay_hugging_face(self, model_url: str, essay: str) -> dict|None:
        """
        Evaluate an essay using the Hugging Face API.

        Args:
            model_url (str): The URL of the Hugging Face model.
            essay (str): The essay to evaluate.

        Returns:
            dict|None: The evaluation results or None if an error occurs.
        """
        prompt = f"""
        ### Instruction:
        Evaluate the essay and return ONLY JSON, no text explanation.
        Scores range from 0 to 9 with step 0.5.

        Return format:
        {format_score_schema()}

        ### Text:
        {essay}

        ### Answer:
        """
        headers = {"Authorization": f"Bearer {self.SERVICE_TOKEN}"}
        messages = [
            {"role": "system", "content": "You are an IELTS examiner. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ]
        payload = {
            "inputs": messages,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.2,
                "do_sample": True,
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

        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            answer = match.group(0)
        else:
            print("JSON не найден:", text)
            return None

        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            print("Ошибка при декодировании:", answer)
            return None

    def get_responses_huggin_face(self, model_url: str, number_of_evaluations: int, 
                                  test_dataset: pd.Series) -> None:
        """
        Get responses for a series of essays using the Hugging Face API.
        
        Args:
            model_url (str): The URL of the Hugging Face model.
            number_of_evaluations (int): The number of evaluations to perform.
            test_dataset (pd.Series): The series of essays to evaluate.

        Returns:
            None
        """
        for evaluation in range(number_of_evaluations):
            results = []
            for i, essay in enumerate(test_dataset):
                result = self.__evaluate_essay_hugging_face(model_url=model_url, essay=essay)
                results.append(result)
                print(f"Training {evaluation+1}: essay {i+1}")
            output_dir = PATHS["results"]["llm_hugging_face_responses"]
            os.makedirs(output_dir, exist_ok=True)
            filename = PATHS["files"]["llm_response"].format(evaluation=evaluation + 1)
            with open(os.path.join(output_dir, filename),
                      "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    #OpenAi
    def __connect_to_openai(self, model_url: str) -> None:
        """
        Connect to the OpenAI API using the provided model URL.
        
        Args:
            model_url (str): The URL of the OpenAI model to connect to.

        Returns:
            None
        """
        from openai import OpenAI

        self.client = OpenAI(
            base_url=model_url,
            api_key=self.SERVICE_TOKEN
        )
        
    def __evaluate_essay_openai(self, essay: str):
        """
        Evaluate an essay using the OpenAI API.

        Args:
            essay (str): The essay to evaluate.

        Returns:
            dict|None: The evaluation results or None if an error occurs.
        """
        from openai.types.chat import ChatCompletionMessageParam

        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": (
                    "You are an IELTS examiner. Return ONLY valid JSON using this schema: "
                    f"{format_score_schema()}"
                ),
            },
            {"role": "user", "content": essay}
        ]

        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=messages,
                max_tokens=200,
                temperature=0.2
            )
        except Exception as e:
            print("Ошибка запроса:", e)
            return None

        text = response.choices[0].message.content
        if text is None:
            return None
        match = re.search(r"\{[\s\S]*?\}", text)
        if not match:
            print("JSON не найден:", text)
            return None

        answer = match.group(0)
        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            print("Ошибка при декодировании:", answer)
            return None
    
    def get_responses_openai(self, model_url: str, number_of_evaluations: int, 
                                  test_dataset: pd.Series) -> None:
        """
        Get responses for a series of essays using the OpenAI API.

        Args:
            model_url (str): The URL of the OpenAI model.
            number_of_evaluations (int): The number of evaluations to perform.
            test_dataset (pd.Series): The series of essays to evaluate.

        Returns:
            None
        """
        self.__connect_to_openai(model_url=model_url)
        for evaluation in range(number_of_evaluations):
            results = []
            for i, essay in enumerate(test_dataset):
                result = self.__evaluate_essay_openai(essay=essay)
                results.append(result)
                print(f"Training {evaluation+1}: essay {i+1}")
            output_dir = PATHS["results"]["llm_openai_responses"]
            os.makedirs(output_dir, exist_ok=True)
            filename = PATHS["files"]["llm_response"].format(evaluation=evaluation + 1)
            with open(os.path.join(output_dir, filename),
                      "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

