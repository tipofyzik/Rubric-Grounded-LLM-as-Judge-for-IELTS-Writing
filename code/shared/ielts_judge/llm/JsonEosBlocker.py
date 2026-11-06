"""Prevent EOS while an IELTS score object is incomplete."""

import json
from typing import cast

import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase

from ielts_judge.Criteria import CRITERION_IDS


class JsonEosBlocker(LogitsProcessor):
    """Block configured EOS tokens until the generated score JSON is complete."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompt_length: int,
                 eos_tokens: list[int],
                 criterion_ids: tuple[str, ...] = CRITERION_IDS) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            tokenizer (PreTrainedTokenizerBase): Tokenizer used by this operation.
            prompt_length (int): Prompt length used by this operation.
            eos_tokens (list[int]): Eos tokens used by this operation.
            criterion_ids (tuple[str, ...]): Criterion ids used by this operation.

        Returns:
            None: This operation returns no value.
        """
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.eos_tokens = eos_tokens
        self.criterion_ids = criterion_ids

    def __call__(self, input_ids: torch.Tensor,
                 scores: torch.Tensor) -> torch.FloatTensor:
        """Suppress EOS independently for each incomplete generated response.

        Args:
            input_ids (torch.Tensor): Input ids used by this operation.
            scores (torch.Tensor): Four criterion scores using the public IELTS score contract.

        Returns:
            torch.FloatTensor: The computed result for the supplied inputs.
        """
        if not self.eos_tokens:
            return cast(torch.FloatTensor, scores)
        for sequence_index, sequence_tokens in enumerate(input_ids):
            # Each slice is one sequence; decode returns a string for this input.
            response = cast(str, self.tokenizer.decode(
                sequence_tokens[self.prompt_length:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ))
            if not self._is_complete(response):
                scores[sequence_index, self.eos_tokens] = -torch.inf
        return cast(torch.FloatTensor, scores)

    def _is_complete(self, response: str) -> bool:
        """Recognize a complete JSON object containing all numeric score fields.

        Args:
            response (str): Response used by this operation.

        Returns:
            bool: The computed result for the supplied inputs.
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
            for criterion in self.criterion_ids
        )
