from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
import json
import re
import os

from huggingface_hub import login
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import requests



class LLMResponseObtainer:
    """
    A class to obtain responses from LLMs for essay evaluation.
    """
    def __init__(self, SERVICE_TOKEN: str) -> None:
        """
        Initialize the LLMResponseObtainer with the necessary service token.

        Args:
            SERVICE_TOKEN (str): The API token for accessing the LLM service.
        """
        self.SERVICE_TOKEN = SERVICE_TOKEN

    # Process locally
    def load_model_and_tokenizer(self, model_url: str):
        """
        Load the model and tokenizer for local evaluation.

        Args:
            model_url (str): The URL or identifier of the model to load.
        """
        login(self.SERVICE_TOKEN)
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_url,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def __evaluate_essay_locally(self, essay: str):
        """
        Evaluate an essay locally using the loaded model.

        Args:
            essay (str): The essay to evaluate.

        Returns:
            dict|None: The evaluation results or None if an error occurs.
        """
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
        """
        messages = [
            {"role": "system", "content": "You are an IELTS examiner. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.2,
                do_sample=True
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Answer:")[-1].strip()
        match = re.search(r"\{[\s\S]*?\}", response)
        if match:
            answer = match.group(0)
        else:
            print("JSON не найден:", response)
            return None
        
        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            print("Ошибка при декодировании:", answer)
            return None

    def get_response_locally(self, number_of_evaluations: int, 
                                   test_dataset: pd.Series) -> None:
        """
        Get responses for a series of essays using the locally loaded model.

        Args:
            test_llm_dataset (pd.Series): The series of essays to evaluate.

        Returns:
            list: A list of evaluation results.
        """
        for evaluation in range(number_of_evaluations):
            results = []
            for i, essay in enumerate(test_dataset):
                result = self.__evaluate_essay_locally(essay)
                results.append(result)
                print(f"Training {evaluation+1}: essay {i+1}")
            os.makedirs("./llm_test_results/hugging_face", exist_ok=True)
            with open(f"./llm_test_results/hugging_face/result_{evaluation+1}.json", 
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
        {{
        "task_achievement": number,
        "coherence_and_cohesion": number,
        "lexical_resource": number,
        "grammatical_range_and_accuracy": number
        }}

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

        response = requests.post(model_url, headers=headers, json=payload)
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
            os.makedirs("./llm_test_results/hugging_face", exist_ok=True)
            with open(f"./llm_test_results/hugging_face/result_{evaluation+1}.json", 
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
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are an IELTS examiner. Return ONLY valid JSON."},
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
            os.makedirs("./llm_test_results/openai", exist_ok=True)
            with open(f"./llm_test_results/openai/result_{evaluation+1}.json", 
                      "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

