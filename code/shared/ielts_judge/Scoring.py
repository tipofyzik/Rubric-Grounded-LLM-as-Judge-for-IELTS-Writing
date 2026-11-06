"""UI-independent scoring orchestration with one bounded prompt/judge/parser chain."""
from typing import Any

from dataclasses import dataclass, replace
import hashlib
import json
import math
import threading
import time
from uuid import uuid4

from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context

from ielts_judge.Criteria import format_rubric
from ielts_judge.Database import EncodingError, RetrievalError
from ielts_judge.Prompts import CONTEXT_SETUPS, JudgeContext, PROMPT_VERSION, build_judge_messages, submission_parts
from ielts_judge.Scores import OUTPUT_SCHEMA_VERSION, parse_scores, public_scores


class ScoringError(RuntimeError):
    """Carry a public failure code and message from a scoring stage."""
    def __init__(self, code: str, message: str) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            code (str): Code used by this operation.
            message (str): Message used by this operation.

        Returns:
            None: This operation returns no value.
        """
        super().__init__(message)
        self.code = code


class JudgeQuotaError(RuntimeError):
    """The hosting adapter recognized a provider quota rejection."""


@dataclass(frozen=True)
class ScoringLimits:
    """Define validated character and elapsed-time limits for one scoring request."""
    timeout_seconds: float = 120
    max_input_chars: int = 40000

    def __post_init__(self) -> None:
        """Validate configuration values before the instance is used.

        Returns:
            None: This operation returns no value.
        """
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if type(self.max_input_chars) is not int or self.max_input_chars < 1:
            raise ValueError("max_input_chars must be a positive integer")


class ScoringService:
    """Runtime supplies loaded components; request data is never persisted or cached.

    One synchronous request owns the runtime. Deadlines are checked between stages
    and between HF decode steps; in-flight native CPU/GPU kernels are not preempted.
    """

    def __init__(
        self,
        judge: Any,
        *,
        aes: Any=None,
        retriever: Any=None,
        setup: str="rubric_panda_aes",
        limits: ScoringLimits | None=None,
        versions: dict[str, Any] | None=None
    ) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            judge (Any): Loaded judge adapter providing token counting and generation.
            aes (Any): Loaded essay-only AES predictor or an explicitly disabled component.
            retriever (Any): Reference adapter using the accepted corpus and embedding profile.
            setup (str): Context setup selecting rubric, PANDA references and AES guidance.
            limits (ScoringLimits | None): Validated request limits for input size and elapsed processing time.
            versions (dict[str, Any] | None): Safe component provenance included in result metadata.

        Returns:
            None: This operation returns no value.
        """
        if setup not in CONTEXT_SETUPS:
            raise ValueError("Unknown context setup")
        self.judge, self.aes, self.retriever, self.setup = judge, aes, retriever, setup
        self.limits = limits if limits is not None else ScoringLimits()
        self.versions = dict(versions or {})
        self._slot = threading.BoundedSemaphore(1)
        self.chain = (
            RunnableLambda(self._prompt)
            | RunnableLambda(self._generate)
            | RunnableLambda(self._parse)
        )

    def _prompt(self, request: dict[str, Any]) -> Any:
        """Build the bounded judge prompt using the current request context.

        Args:
            request (dict[str, Any]): Request-local state owned by the synchronous scoring operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        def prepare() -> Any:
            """Assemble the current operation's validated inputs and metadata.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            context = request["context"]
            references = context.references
            dropped = []
            limit = (
                request["profile"].context_budget - request["profile"].max_new_tokens
            )
            while True:
                messages = build_judge_messages(
                    request["task_prompt"], request["essay"], context
                )
                count = self.judge.count_input_tokens(messages)
                if count <= limit:
                    break
                if not context.references:
                    raise ScoringError(
                        "input_too_long",
                        "Task/Essay and required prompt exceed the token budget",
                    )
                dropped.append(context.references[-1].essay_id)
                context = replace(context, references=context.references[:-1])
                if not context.references:
                    raise ScoringError(
                        "context_budget_exceeded",
                        "No complete reference fits the judge token budget",
                    )
            request["messages"] = messages
            request["input_tokens"] = count
            request["metadata"]["prompt"] = {
                "version": PROMPT_VERSION,
                "input_tokens": count,
                "input_limit": limit,
                "sha256": hashlib.sha256(
                    json.dumps(messages, ensure_ascii=False).encode()
                ).hexdigest(),
            }
            request["metadata"]["retrieval"].update(
                used_reference_ids=[reference.essay_id for reference in context.references],
                dropped_reference_ids=dropped,
                found_count=len(references),
                used_count=len(context.references),
            )
            if dropped:
                request["warnings"].append("references_reduced_for_budget")
            return request

        return request["run"]("prompt", prepare)

    def _generate(self, request: dict[str, Any]) -> Any:
        """Invoke the selected judge once and verify the returned component provenance.

        Args:
            request (dict[str, Any]): Request-local state owned by the synchronous scoring operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        request["metadata"]["judge_attempts"] = 1
        result = request["run"](
            "judge",
            lambda: self.judge.predict_messages(
                model_key=request["model_key"],
                messages=request["messages"],
                deadline=request["deadline"],
            ),
        )
        profile = request["profile"]
        expected = {
            "model_id": profile.repo_id,
            "model_revision": profile.revision,
            "prompt_version": PROMPT_VERSION,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "input_tokens": request["input_tokens"],
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise ScoringError(
                "judge_metadata_mismatch",
                "Judge response does not match the prepared runtime/prompt",
            )
        request["generation"] = result
        return request

    def _parse(self, request: dict[str, Any]) -> Any:
        """Validate the judge completion and attach four public integer criterion scores.

        Args:
            request (dict[str, Any]): Request-local state owned by the synchronous scoring operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        request["scores"] = request["run"](
            "parser", lambda: parse_scores(request["generation"]["raw_text"])
        )
        return request

    def score(self, *, model_key: str, task_prompt: str, essay: str, essay_id: str | None=None) -> dict[str, Any]:
        """Evaluate one Task/Essay submission and return a bounded success or failure result.

        Args:
            model_key (str): Enabled immutable judge profile key from the model registry.
            task_prompt (str): English Task 2 question against which the essay is assessed.
            essay (str): English essay text supplied for evaluation.
            essay_id (str | None): Stable source essay identifier used for reference exclusion.

        Returns:
            dict[str, Any]: Status, four validated criterion scores or an error, and safe processing metadata.
        """
        started = time.monotonic()
        stage = "validation"
        timings = {}
        rubric, use_references, use_aes = CONTEXT_SETUPS[self.setup]
        metadata = {
            "versions": dict(self.versions),
            "context_setup": self.setup,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "rubric_sha256": hashlib.sha256(format_rubric().encode()).hexdigest(),
            "aes": {"status": "not_run" if use_aes else "disabled"},
            "retrieval": {"status": "not_run" if use_references else "disabled"},
            "judge_attempts": 0,
        }
        warnings = []
        result = {
            "status": "error",
            "run_id": uuid4().hex,
            "model_key": model_key,
            "model_id": None,
            "metadata": metadata,
            "timings": timings,
        }
        if not self._slot.acquire(blocking=False):
            result["error"] = {
                "code": "service_busy",
                "stage": "admission",
                "message": "Scoring runtime is busy",
            }
            timings["total_seconds"] = time.monotonic() - started
            return result
        deadline = started + self.limits.timeout_seconds

        def run(name: Any, operation: Any) -> Any:
            """Execute one named stage under the original request deadline and record its duration.

            Args:
                name (Any): Name used by this operation.
                operation (Any): Operation used by this operation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            nonlocal stage
            stage = name
            if time.monotonic() >= deadline:
                raise TimeoutError("Request deadline expired")
            tick = time.monotonic()
            try:
                value = operation()
            finally:
                timings[name + "_seconds"] = time.monotonic() - tick
            if time.monotonic() >= deadline:
                raise TimeoutError("Request deadline expired")
            return value

        try:

            def validate() -> Any:
                """Validate the current input before any dependent work is performed.

                Returns:
                    Any: The computed result for the supplied inputs.
                """
                profile = self.judge.registry.require_enabled(model_key)
                if profile.key != self.judge.profile.key:
                    raise ValueError("Selected model is not loaded")
                submission_parts({"task_prompt": task_prompt, "essay": essay})
                if len(task_prompt) + len(essay) > self.limits.max_input_chars:
                    raise ScoringError(
                        "input_too_long",
                        "Submission exceeds the server character limit",
                    )
                if essay_id is not None and (
                    not isinstance(essay_id, str) or not essay_id.strip()
                ):
                    raise ValueError("essay_id must be nonempty or None")
                return profile

            profile = run("validation", validate)
            result["model_id"] = profile.repo_id
            metadata["model_revision"] = profile.revision
            context = JudgeContext(include_rubric=rubric)
            # Reject impossible submissions before running the other models or DB.
            preflight = build_judge_messages(task_prompt, essay, context)
            if (
                run("preflight", lambda: self.judge.count_input_tokens(preflight))
                > profile.context_budget - profile.max_new_tokens
            ):
                raise ScoringError(
                    "input_too_long",
                    "Task/Essay and required prompt exceed the token budget",
                )
            if use_aes:
                aes = self.aes
                metadata["aes"]["status"] = "running"

                def predict_aes() -> Any:
                    """Predict aes.

                    Returns:
                        Any: The computed result for the supplied inputs.
                    """
                    if aes is None:
                        raise ScoringError("aes_unavailable", "AES runtime is not configured")
                    prediction = aes.predict_essay(essay)
                    labels = public_scores(prediction["scores"], legacy=True)
                    return prediction, labels

                prediction, labels = run("aes", predict_aes)
                metadata["aes"] = {
                    "status": "ok",
                    "scores": labels,
                    "input": prediction["input"],
                    "semantics": prediction["semantics"],
                }
                context = replace(context, aes_priors=labels)
            if use_references:
                retriever = self.retriever
                metadata["retrieval"]["status"] = "running"

                def retrieve() -> Any:
                    """Encode the query essay and retrieve compatible training references with ID/hash exclusion.

                    Returns:
                        Any: The computed result for the supplied inputs.
                    """
                    if retriever is None:
                        raise ScoringError("retrieval_unavailable", "Retrieval runtime is not configured")
                    return retriever.retrieve(
                        essay, essay_id=essay_id, timeout=deadline - time.monotonic()
                    )

                found = run("retrieval", retrieve)
                if found["status"] not in ("ok", "retrieval_empty"):
                    raise RetrievalError("Unknown retrieval result status")
                context = replace(context, references=found["references"])
                metadata["retrieval"] = {
                    key: value for key, value in found.items() if key != "references"
                }
                metadata["retrieval"]["reference_ids"] = [
                    current_record.essay_id for current_record in found["references"]
                ]
                if found["status"] == "retrieval_empty":
                    warnings.append("retrieval_empty")
            request = {
                "task_prompt": task_prompt,
                "essay": essay,
                "model_key": model_key,
                "profile": profile,
                "context": context,
                "metadata": metadata,
                "warnings": warnings,
                "run": run,
                "deadline": deadline,
            }
            # Explicitly suppress ambient LangSmith tracing for user submissions.
            with tracing_context(enabled=False):
                request = self.chain.invoke(request, config={"callbacks": []})
            generation = request["generation"]
            metadata["generation"] = {
                key: generation[key]
                for key in (
                    "input_tokens",
                    "generated_tokens",
                    "max_new_tokens",
                    "stop_reason",
                )
            }
            for key in ("gpu_work_seconds", "gpu_peak_allocated_mib", "worker_peak_rss_mib",
                        "allocation_and_transport_seconds", "container_memory_mib",
                        "container_memory_peak_mib"):
                if key in generation:
                    metadata["generation"][key] = generation[key]
            result.update(status="ok", scores=request["scores"], warnings=warnings)
        except Exception as error:
            if isinstance(error, ScoringError):
                code, message = error.code, str(error)
            elif isinstance(error, JudgeQuotaError):
                code, message = "judge_quota", "Judge GPU quota is unavailable"
            elif isinstance(error, EncodingError):
                code, message = "encoder_failure", "Query encoding failed"
            elif isinstance(error, RetrievalError):
                code, message = error.code, "Reference retrieval failed"
            elif isinstance(error, TimeoutError):
                code, message = (
                    stage + "_timeout",
                    "Request deadline or dependency timeout exceeded",
                )
            elif stage == "validation":
                code, message = (
                    "invalid_input",
                    "Model key, Task and Essay must satisfy the input contract",
                )
            elif stage == "parser":
                code, message = (
                    "invalid_output",
                    "Judge did not return four integer scores in 0-9",
                )
            else:
                code, message = stage + "_failure", "Scoring stage failed"
            result["error"] = {"code": code, "stage": stage, "message": message}
            details = error.diagnostics if isinstance(error, RetrievalError) else {}
            result["error"].update(details)
            for component in ("aes", "retrieval"):
                if metadata[component].get("status") == "running":
                    metadata[component].update(status="error", error_code=code, **details)
        finally:
            timings["total_seconds"] = time.monotonic() - started
            self._slot.release()
        return result
