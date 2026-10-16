"""Prevent EOS while an IELTS score object is incomplete."""

import json

import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase

from ..criteria import CRITERION_IDS


class JsonEosBlocker(LogitsProcessor):
    """Block configured EOS tokens until the generated score JSON is complete."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompt_length: int,
                 eos_tokens: list[int]) -> None:
        """Store the decoder and the boundary separating prompt and response.

        Args:
            tokenizer (PreTrainedTokenizerBase): Tokenizer used for generation.
            prompt_length (int): Number of prompt tokens to exclude from parsing.
            eos_tokens (list[int]): Model-configured tokens that end generation.

        Returns:
            None.
        """
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.eos_tokens = eos_tokens

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        """Suppress EOS independently for each incomplete generated response.

        Args:
            input_ids (torch.LongTensor): Prompt and generated tokens per sequence.
            scores (torch.FloatTensor): Next-token scores for each sequence.

        Returns:
            torch.FloatTensor: Scores with EOS blocked for incomplete objects.
        """
        if not self.eos_tokens:
            return scores
        for sequence_index, sequence_tokens in enumerate(input_ids):
            response = self.tokenizer.decode(
                sequence_tokens[self.prompt_length:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not self._is_complete(response):
                scores[sequence_index, self.eos_tokens] = -torch.inf
        return scores

    def _is_complete(self, response: str) -> bool:
        """Recognize a complete JSON object containing all numeric score fields.

        Args:
            response (str): Generated text without the prompt.

        Returns:
            bool: Whether EOS can be allowed for this response.
        """
        object_start = response.find("{")
        if object_start < 0:
            return False
        try:
            answer, _ = json.JSONDecoder().raw_decode(response[object_start:])
        except ValueError:
            return False
        return isinstance(answer, dict) and all(
            isinstance(answer.get(criterion), (int, float))
            and not isinstance(answer[criterion], bool)
            for criterion in CRITERION_IDS
        )
