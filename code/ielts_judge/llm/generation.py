"""Single-response HF generation shared by research and serving callers."""

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ..criteria import CRITERION_IDS
from .JsonEosBlocker import JsonEosBlocker


@dataclass(frozen=True)
class GenerationResult:
    """Completion and diagnostics owned by one call; no parsing or persistence."""

    raw_text: str
    input_tokens: int
    generated_tokens: int
    max_new_tokens: int
    last_token_id: int | None
    stop_reason: str


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    generation_settings: Mapping[str, Any],
    criterion_ids: tuple[str, ...] = CRITERION_IDS,
    input_token_limit: int | None = None,
) -> GenerationResult:
    """Generate once using an already-loaded model, without truncating input.

    Model placement and GPU allocation belong to the calling runtime. This
    function neither loads weights nor retries, logs content, or writes files.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_length = inputs["input_ids"].shape[1]
    if input_token_limit is not None and input_length > input_token_limit:
        raise ValueError(
            f"Judge input has {input_length} tokens; limit is {input_token_limit}."
        )
    inputs = inputs.to(model.device)
    config = getattr(model, "generation_config", None)
    eos_tokens = getattr(config, "eos_token_id", None)
    if eos_tokens is None:
        eos_tokens = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_tokens, int):
        eos_tokens = [eos_tokens]
    eos_tokens = eos_tokens or []
    blocker = JsonEosBlocker(tokenizer, input_length, eos_tokens, criterion_ids)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            logits_processor=[blocker],
            **generation_settings,
        )
    completion = outputs[0, input_length:]
    raw_text = tokenizer.decode(
        completion, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    count = int(completion.numel())
    last_token = int(completion[-1].item()) if count else None
    if last_token is not None and last_token in eos_tokens:
        stop_reason = "eos"
    elif count >= max_new_tokens:
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "other"
    return GenerationResult(
        raw_text, int(input_length), count, max_new_tokens, last_token, stop_reason,
    )
