"""Local hybrid startup and hosted judge transport, shared by CLI and UI."""
from typing import Any

import importlib.metadata
import json
import os
from pathlib import Path
import time

import requests

from ielts_judge.ModelRegistry import ModelRegistry
from ielts_judge.Paths import find_release
from ielts_judge.Prompts import CONTEXT_SETUPS

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())


def configure_environment() -> None:
    """Configure offline Hugging Face access and local caches for this process."""
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_TEMP_DIR"] = str(ROOT / ".cache/gradio")


from ielts_judge.Access import LauncherError, read_access as read_settings


def read_access(setup: str) -> dict[str, str]:
    """Read settings for local scoring, with process values taking precedence.

    Args:
        setup (str): Context setup selecting rubric, PANDA references and AES guidance.

    Returns:
        dict[str, str]: Runtime settings for the selected setup.
    """
    return read_settings(setup, root=ROOT)


def check_space(values: dict[str, str]) -> tuple[str, str]:
    """Discover the remote judge endpoint exposed by the configured Space.

    Args:
        values (dict[str, str]): Settings containing HF_SPACE_ID and HF_TOKEN.

    Returns:
        tuple[str, str]: HTTPS Space origin and the available endpoint name.

    Raises:
        LauncherError: The Space identifier, access, response or endpoint is invalid.
    """
    repository = values.get("HF_SPACE_ID", "")
    import re
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        raise LauncherError("invalid_space", "HF_SPACE_ID must be an owner/name repository ID.")
    base = "https://" + repository.replace("/", "-").lower() + ".hf.space"
    try:
        response = requests.get(base + "/config", headers={"Authorization": "Bearer " + values["HF_TOKEN"]}, timeout=15)
    except requests.RequestException as error:
        raise LauncherError("space_unreachable", "Cannot reach the Space. Check the connection and its HF runtime.") from error
    if response.status_code != 200:
        raise LauncherError("space_unavailable", "The Space is unavailable or access is denied. Check its runtime and HF_TOKEN.")
    try:
        endpoints = {record.get("api_name") for record in response.json().get("dependencies", [])}
    except (ValueError, AttributeError, TypeError) as error:
        raise LauncherError("invalid_space_response", "The Space did not return its API configuration.") from error
    for name in ("judge", "stage5_judge"):
        if name in endpoints:
            return base, name
    raise LauncherError("missing_endpoint", "The running Space does not expose a judge endpoint.")


def preflight(values: dict[str, str], setup: str) -> dict[str, Any]:
    """Check that required local files and the hosted judge endpoint are available.

    Read installed package versions without comparing pins. This check does not
    load models, run inference or query Supabase.

    Args:
        values (dict[str, str]): Settings containing HF_SPACE_ID and HF_TOKEN.
        setup (str): Context setup determining whether AES and PANDA files are required.

    Returns:
        dict[str, Any]: Readiness status, Space origin, endpoint and installed versions.
    """
    _, references, aes = CONTEXT_SETUPS[setup]
    if aes:
        path = find_release(ROOT, "aes")
        release = json.loads(path.read_text(encoding="utf-8"))
        bundle = path.parent / release["bundle_path"]
        if not (bundle / "manifest.json").is_file() or not (bundle / "ordinal_model.pt").is_file():
            raise LauncherError("missing_aes", "The accepted AES bundle is missing.")
    if references:
        release = json.loads(find_release(ROOT, "panda").read_text(encoding="utf-8"))
        corpus = ROOT / release["corpus_path"]
        if not (corpus / "manifest.json").is_file() or not (corpus / "records.jsonl").is_file():
            raise LauncherError("missing_corpus", "The accepted corpus artifacts are missing.")
    packages = {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "huggingface-hub", "sentence-transformers", "langchain-core", "langsmith")}
    space, endpoint = check_space(values)
    return {"status": "ready", "mode": "preflight", "context_setup": setup,
            "space": space, "judge_endpoint": endpoint, "packages": packages,
            "loads_models": False, "runs_inference": False,
            "scope": "Local artifact presence, installed packages and hosted endpoint availability; no DB/model inference check."}


class RemoteJudge:
    """Adapt the hosted judge endpoint to the shared scoring interface and tokenizer contract."""
    def __init__(self, values: Any, base: Any, endpoint: str = "judge") -> None:
        """Load the pinned judge tokenizer from the local cache and configure remote calls.

        Args:
            values (Any): Settings with HF_TOKEN and an optional IELTS_MODEL_KEY.
            base (Any): HTTPS origin of the Space.
            endpoint (str): Endpoint name discovered in the Space configuration.
        """
        self.registry = ModelRegistry()
        self.profile = self.registry.require_enabled(values.get("IELTS_MODEL_KEY", "qwen25_1p5b"))
        self.base = f"{base}/gradio_api/call/{endpoint}"
        self.headers = {"Authorization": "Bearer " + values["HF_TOKEN"]}
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.profile.repo_id, revision=self.profile.revision, local_files_only=True,
            clean_up_tokenization_spaces=False)

    def count_input_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count tokens after applying the complete judge chat template.

        Args:
            messages (list[dict[str, str]]): Ordered chat messages containing role and content.

        Returns:
            int: Token count, including the generation prompt.
        """
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return int(self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])

    def predict_messages(self, *, model_key: str, messages: list[dict[str, str]], deadline: float) -> Any:
        """Submit a judge request and read the first completed Gradio event.

        Args:
            model_key (str): Enabled judge profile key from the model registry.
            messages (list[dict[str, str]]): Ordered chat messages containing role and content.
            deadline (float): Absolute monotonic deadline, in seconds.

        Returns:
            Any: First item of the completion event, as returned by the hosted judge.

        Raises:
            TimeoutError: The shared deadline or an HTTP timeout expires.
            JudgeQuotaError: The hosted endpoint reports GPU quota exhaustion.
            RuntimeError: The stream reports an error or ends without a completion.
        """
        from ielts_judge.Scoring import JudgeQuotaError
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Request deadline expired")
        payload = {"model_key": model_key, "messages": messages, "max_seconds": min(55, remaining)}
        try:
            with requests.Session() as session:
                submitted = session.post(self.base, headers=self.headers, json={"data": [payload]}, timeout=min(15, remaining))
                submitted.raise_for_status()
                event_id = submitted.json()["event_id"]
                event = None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Request deadline expired")
                with session.get(self.base + "/" + event_id, headers=self.headers, stream=True,
                                 timeout=(min(10, remaining), remaining)) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Request deadline expired")
                        if line.startswith("event:"):
                            event = line.split(":", 1)[1].strip()
                        if line.startswith("data:") and event == "complete":
                            return json.loads(line.split(":", 1)[1].strip())[0]
                        if line.startswith("data:") and event == "error":
                            if "quota" in line.lower():
                                raise JudgeQuotaError("Hosted GPU quota exceeded")
                            raise RuntimeError("Hosted judge returned an error")
        except requests.Timeout as error:
            raise TimeoutError("Hosted judge request timed out") from error
        raise RuntimeError("Hosted stream ended without a result")


def load_service(judge: Any, values: Any, setup: str) -> Any:
    """Load the local scoring components required by the selected context setup.

    Args:
        judge (Any): Remote judge adapter providing token counting and generation.
        values (Any): Settings with Supabase URL and publishable key when PANDA is enabled.
        setup (str): Context setup selecting rubric, PANDA references and AES guidance.

    Returns:
        Any: ScoringService configured with the local components and remote judge.
    """
    import torch
    from transformers.utils import logging
    logging.set_verbosity_error()
    from ielts_judge.ScoringRuntime import load_scoring_service
    torch.set_num_threads(4)
    return load_scoring_service(judge, aes_release=find_release(ROOT, "aes"),
        corpus_release=find_release(ROOT, "panda"), project_root=ROOT,
        supabase_url=values.get("SUPABASE_URL"), supabase_key=values.get("SUPABASE_PUBLISHABLE_KEY"), setup=setup)


