"""Public single-submission judge call, independent of UI and research writers."""

from dataclasses import asdict
from typing import Any

from ..criteria import PUBLIC_CRITERION_IDS
from ..model_registry import ModelRegistry
from ..prompts import JudgeContext, PROMPT_VERSION, build_judge_messages
from .generation import generate_response


class Judge:
    """One prepared model profile; runtime owns loading, placement and concurrency.

    Attach preloaded weights at runtime startup, including ZeroGPU startup.
    Attached model/tokenizer must correspond to the selected profile; exposed
    commit metadata is checked when available. Predict never swaps models.
    """

    def __init__(self, model_key: str, model: Any, tokenizer: Any,
                 registry: ModelRegistry | None = None) -> None:
        self.registry = registry if registry is not None else ModelRegistry()
        self.profile = self.registry.require_enabled(model_key)
        model_commit = getattr(getattr(model, "config", None), "_commit_hash", None)
        tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        for commit in (model_commit, tokenizer_commit):
            if isinstance(commit, str) and commit != self.profile.revision:
                raise ValueError("Loaded model/tokenizer revision differs from judge profile.")
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_key: str, *, device: str = "cpu", token: str | None = None,
                        registry: ModelRegistry | None = None) -> "Judge":
        """Explicit startup loader; never use this inside a request callback.

        ZeroGPU callers select device='cuda' at module startup and wrap predict
        with spaces.GPU. This module has no dependency on spaces or Gradio.
        """
        registry = registry if registry is not None else ModelRegistry()
        profile = registry.require_enabled(model_key)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            profile.repo_id, revision=profile.revision, token=token,
            clean_up_tokenization_spaces=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            profile.repo_id, revision=profile.revision, token=token,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        )
        model.to(device)
        return cls(model_key, model, tokenizer, registry)

    def predict(self, *, model_key: str, task_prompt: str, essay: str,
                context: JudgeContext | None = None) -> dict:
        """Return completion/metadata, not validated scores; no content logging.

        The later ScoringService owns parsing and score validation. Model/GPU
        failures propagate to that caller without retries or score imputation.
        """
        profile = self.registry.require_enabled(model_key)
        if profile.key != self.profile.key:
            raise ValueError(f"Judge has {self.profile.key} loaded, not {model_key}.")
        messages = build_judge_messages(task_prompt, essay, context)
        generation = generate_response(
            self.model, self.tokenizer, messages,
            max_new_tokens=profile.max_new_tokens,
            generation_settings=profile.generation_config,
            criterion_ids=PUBLIC_CRITERION_IDS,
            input_token_limit=profile.context_budget - profile.max_new_tokens,
        )
        return {
            **asdict(generation), "model_key": profile.key,
            "model_id": profile.repo_id, "model_revision": profile.revision,
            "output_schema_version": profile.output_schema_version,
            "prompt_version": PROMPT_VERSION,
        }
