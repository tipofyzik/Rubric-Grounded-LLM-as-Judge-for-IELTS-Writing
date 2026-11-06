"""Shared local Hugging Face model loading for PANDA components."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class PandaModel:
    """Manage panda model through the operations defined by this component."""
    def __init__(self, service_token: str) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            service_token (str): Service token used by this operation.

        Returns:
            None: This operation returns no value.
        """
        self.service_token = service_token

    def load_model_and_tokenizer(self, model_url: str, *, revision: str | None = None) -> None:
        """Load the model and tokenizer for local evaluation.

        Args:
            model_url (str): Model url used by this operation.
            revision (str | None): Revision used by this operation.

        Returns:
            None: This operation returns no value.
        """
        token = self.service_token or None
        self.tokenizer = AutoTokenizer.from_pretrained(model_url, revision=revision, token=token)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_url,
            revision=revision,
            token=token,
            dtype=torch.float16,
            device_map="auto"
        )
