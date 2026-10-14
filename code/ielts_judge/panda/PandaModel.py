"""Shared local Hugging Face model loading for PANDA components."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class PandaModel:
    def __init__(self, service_token: str) -> None:
        """Store the optional token used when loading a gated model.

        Args:
            service_token (str): Hugging Face token, or an empty string for public models.
        """
        self.service_token = service_token

    def load_model_and_tokenizer(self, model_url: str) -> None:
        """
        Load the model and tokenizer for local evaluation.

        Args:
            model_url (str): The URL or identifier of the model to load.
        """
        if self.service_token:
            from huggingface_hub import login

            login(self.service_token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_url,
            dtype=torch.float16,
            device_map="auto"
        )
