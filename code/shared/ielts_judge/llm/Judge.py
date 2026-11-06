"""Public single-submission judge call, independent of UI and research writers."""

from dataclasses import asdict
from typing import Any, cast

from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from ielts_judge.ModelRegistry import ModelRegistry
from ielts_judge.Prompts import JudgeContext, PROMPT_VERSION, build_judge_messages
from ielts_judge.llm.Generation import generate_response


class Judge:
    """One prepared model profile; runtime owns loading, placement and concurrency.

    Attach preloaded weights at runtime startup, including ZeroGPU startup.
    Attached model/tokenizer must correspond to the selected profile; exposed
    commit metadata is checked when available. Predict never swaps models.
    """

    def __init__(self, model_key: str, model: Any, tokenizer: Any,
                 registry: ModelRegistry | None = None) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            model_key (str): Enabled immutable judge profile key from the model registry.
            model (Any): Model used by this operation.
            tokenizer (Any): Tokenizer used by this operation.
            registry (ModelRegistry | None): Registry used by this operation.

        Returns:
            None: This operation returns no value.
        """
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

        Args:
            model_key (str): Enabled immutable judge profile key from the model registry.
            device (str): Device used by this operation.
            token (str | None): Token used by this operation.
            registry (ModelRegistry | None): Registry used by this operation.

        Returns:
            'Judge': The computed result for the supplied inputs.
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
        cast(torch.nn.Module, model).to(device)
        return cls(model_key, model, tokenizer, registry)

    def predict(self, *, model_key: str, task_prompt: str, essay: str,
                context: JudgeContext | None = None) -> dict:
        """Return completion/metadata, not validated scores; no content logging.

        The later ScoringService owns parsing and score validation. Model/GPU
        failures propagate to that caller without retries or score imputation.

        Args:
            model_key (str): Enabled immutable judge profile key from the model registry.
            task_prompt (str): English Task 2 question against which the essay is assessed.
            essay (str): English essay text supplied for evaluation.
            context (JudgeContext | None): Context used by this operation.

        Returns:
            dict: Named result fields and metadata produced by this operation.
        """
        self.registry.require_enabled(model_key)
        messages = build_judge_messages(task_prompt, essay, context)
        return self.predict_messages(model_key=model_key, messages=messages)

    def count_input_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count exactly the chat rendering used by generate_response, without truncation.

        Args:
            messages (list[dict[str, str]]): Ordered chat messages rendered with the selected tokenizer template.

        Returns:
            int: The computed result for the supplied inputs.
        """
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return int(self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])

    def predict_messages(self, *, model_key: str, messages: list[dict[str, str]],
                         deadline: float | None = None) -> dict:
        """Generate from the already-budgeted prompt; loading and GPU remain runtime-owned.

        Args:
            model_key (str): Enabled immutable judge profile key from the model registry.
            messages (list[dict[str, str]]): Ordered chat messages rendered with the selected tokenizer template.
            deadline (float | None): Absolute monotonic request deadline shared by the processing stages.

        Returns:
            dict: Named result fields and metadata produced by this operation.
        """
        profile = self.registry.require_enabled(model_key)
        if profile.key != self.profile.key:
            raise ValueError(f"Judge has {self.profile.key} loaded, not {model_key}.")
        generation = generate_response(
            self.model, self.tokenizer, messages,
            max_new_tokens=profile.max_new_tokens,
            generation_settings=profile.generation_config,
            criterion_ids=PUBLIC_CRITERION_IDS,
            input_token_limit=profile.context_budget - profile.max_new_tokens,
            deadline=deadline,
        )
        return {
            **asdict(generation), "model_key": profile.key,
            "model_id": profile.repo_id, "model_revision": profile.revision,
            "output_schema_version": profile.output_schema_version,
            "prompt_version": PROMPT_VERSION,
        }
