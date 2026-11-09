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
        """Bind preloaded weights and tokenizer to an enabled judge profile.

        Args:
            model_key (str): Enabled judge profile key.
            model (Any): Preloaded causal language model for the pinned revision.
            tokenizer (Any): Tokenizer corresponding to the same revision.
            registry (ModelRegistry | None): Registry override, or None for the shared registry.

        Raises:
            ValueError: Exposed model or tokenizer commit metadata differs from the profile.
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
        """Load the pinned model and tokenizer at runtime startup.

        ZeroGPU callers load on CUDA before serving and schedule prediction with
        spaces.GPU.

        Args:
            model_key (str): Enabled judge profile key.
            device (str): Target PyTorch device. CPU uses float32, other devices use float16.
            token (str | None): Optional Hugging Face access token for loading the artifacts.
            registry (ModelRegistry | None): Registry override, or None for the shared registry.

        Returns:
            'Judge': Ready judge bound to the selected model and device.
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
        """Build a prompt and generate one raw judge completion.

        ScoringService parses and validates scores. Generation failures propagate
        to the caller.

        Args:
            model_key (str): Enabled key matching the loaded judge.
            task_prompt (str): IELTS Writing Task 2 question.
            essay (str): Essay to score.
            context (JudgeContext | None): Optional rubric, reference insights and AES guidance.

        Returns:
            dict: Raw completion, token and stop diagnostics, and model and prompt versions.
        """
        self.registry.require_enabled(model_key)
        messages = build_judge_messages(task_prompt, essay, context)
        return self.predict_messages(model_key=model_key, messages=messages)

    def count_input_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count tokens for the same chat rendering used in generation.

        Args:
            messages (list[dict[str, str]]): Ordered chat messages with role and content.

        Returns:
            int: Token count with the generation prompt, before truncation.
        """
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return int(self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])

    def predict_messages(self, *, model_key: str, messages: list[dict[str, str]],
                         deadline: float | None = None) -> dict:
        """Generate from prepared chat messages using the loaded profile.

        Args:
            model_key (str): Enabled key matching the loaded judge.
            messages (list[dict[str, str]]): Ordered chat messages with role and content.
            deadline (float | None): Optional absolute monotonic deadline in seconds.

        Returns:
            dict: Raw completion, token and stop diagnostics, and model and prompt versions.

        Raises:
            ValueError: The requested model differs from the loaded profile.
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
