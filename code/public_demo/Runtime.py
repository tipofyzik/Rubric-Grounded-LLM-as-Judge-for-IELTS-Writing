"""Pinned private artifacts and the CPU-to-ZeroGPU judge boundary."""
from typing import Any

import hashlib
import json
import math
import os
from pathlib import Path
import time


def file_hash(path: Path) -> str:
    """Calculate a file's SHA-256 digest for artifact and deployment verification.
    
    Args:
        path (Path): Filesystem location to read, validate or write.
    
    Returns:
        str: The computed result for the supplied inputs.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_artifacts(root: Path, deployment: dict[str, Any]) -> Path:
    """Restore one immutable release; never obtain artifacts from a request.
    
    Args:
        root (Path): Project root used to resolve relative configuration paths.
        deployment (dict[str, Any]): Deployment used by this operation.
    
    Returns:
        Path: The computed result for the supplied inputs.
    """
    from huggingface_hub import snapshot_download

    target = Path(root) / ".runtime/artifacts"
    revision = deployment["artifact_revision"]
    if len(revision) != 40 or any(current_criterion not in "0123456789abcdef" for current_criterion in revision):
        raise ValueError("Artifacts require an immutable Hub commit")
    snapshot_download(
        deployment["artifact_repo"], revision=revision, repo_type="model",
        token=os.environ["HF_TOKEN"], local_dir=target,
        allow_patterns=["ARTIFACT_MANIFEST.json", "results/**"],
    )
    manifest_path = target / "ARTIFACT_MANIFEST.json"
    if file_hash(manifest_path) != deployment["artifact_manifest_sha256"]:
        raise ValueError("Private artifact manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (target / name).resolve()
        if not path.is_relative_to(target.resolve()) or file_hash(path) != expected:
            raise ValueError("Private artifact integrity failure")
    return target


def validate_payload(payload: dict[str, Any], model_key: str) -> float:
    """Validate the judge request and its optional deadline.
    
    Args:
        payload (dict[str, Any]): Payload used by this operation.
        model_key (str): Enabled immutable judge profile key from the model registry.
    
    Returns:
        float: The computed result for the supplied inputs.
    """
    required = {"model_key", "messages", "max_seconds"}
    if not isinstance(payload, dict) or set(payload) not in (required, required | {"deadline"}):
        raise ValueError("Invalid judge payload")
    if payload["model_key"] != model_key:
        raise ValueError("Model is not loaded")
    messages = payload["messages"]
    if (not isinstance(messages, list) or len(messages) != 2
            or any(not isinstance(current_match, dict) or set(current_match) != {"role", "content"}
                   or not isinstance(current_match["content"], str) for current_match in messages)
            or [current_match["role"] for current_match in messages] != ["system", "user"]
            or sum(len(current_match["content"]) for current_match in messages) > 60000):
        raise ValueError("Invalid prepared messages")
    seconds = payload["max_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 55:
        raise ValueError("Invalid generation duration")
    deadline = payload.get("deadline", time.monotonic() + seconds)
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("Invalid generation deadline")
    deadline = min(deadline, time.monotonic() + seconds)
    if deadline <= time.monotonic():
        raise TimeoutError("Judge deadline expired while waiting for GPU")
    return deadline


class HostedJudge:
    """Adapt a ZeroGPU generation callback to the shared scoring interface."""

    def __init__(self, judge: Any, generate: Any) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            judge (Any): Loaded judge adapter providing token counting and generation.
            generate (Any): Generate used by this operation.

        Returns:
            None: This operation returns no value.
        """
        self.registry, self.profile = judge.registry, judge.profile
        self.tokenizer = judge.tokenizer
        self._judge, self._generate = judge, generate

    def count_input_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count the complete chat render with the selected judge tokenizer.

        Args:
            messages (list[dict[str, str]]): Ordered chat messages rendered with the selected tokenizer template.

        Returns:
            int: The computed result for the supplied inputs.
        """
        return self._judge.count_input_tokens(messages)

    def predict_messages(self, *, model_key: str, messages: list[dict[str, str]], deadline: float) -> Any:
        """Generate a completion for validated chat messages within the shared deadline.

        Args:
            model_key (str): Enabled immutable judge profile key from the model registry.
            messages (list[dict[str, str]]): Ordered chat messages rendered with the selected tokenizer template.
            deadline (float): Absolute monotonic request deadline shared by the processing stages.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from ielts_judge.Scoring import JudgeQuotaError

        started = time.monotonic()
        remaining = min(55, deadline - started)
        if remaining <= 0:
            raise TimeoutError("Judge deadline expired")
        try:
            result = self._generate({"model_key": model_key, "messages": messages,
                                     "max_seconds": remaining, "deadline": min(deadline, started + remaining)})
        except Exception as error:
            # spaces raises gr.Error for scheduler quota rejections. Preserve all
            # other failures, and never expose provider payloads to the user.
            title = str(getattr(error, "title", "")).lower()
            if "quota" in str(error).lower() or "quota" in title:
                raise JudgeQuotaError("ZeroGPU quota unavailable") from error
            if str(error) == "TimeoutError" or "timeout" in title:
                raise TimeoutError("ZeroGPU generation deadline expired") from error
            raise
        result["allocation_and_transport_seconds"] = max(
            0, time.monotonic() - started - result["gpu_work_seconds"])
        # Container-level RAM includes the web, CPU and GPU worker processes.
        for filename, key in (("memory.current", "container_memory_mib"),
                              ("memory.peak", "container_memory_peak_mib")):
            path = Path("/sys/fs/cgroup") / filename
            try:
                if path.is_file():
                    result[key] = int(path.read_text().strip()) / 1024**2
            except (OSError, ValueError):
                pass  # Optional host metric; unavailable is not a scoring failure.
        return result
