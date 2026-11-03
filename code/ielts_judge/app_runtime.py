"""Local hybrid startup and hosted judge transport, shared by CLI and UI."""

import importlib.metadata
import json
import os
from pathlib import Path
import time

import requests

from .model_registry import ModelRegistry
from .prompts import CONTEXT_SETUPS

ROOT = Path(__file__).resolve().parents[2]


def configure_environment():
    """Select the accepted local cache before importing model libraries."""
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_TEMP_DIR"] = str(ROOT / ".cache/gradio")


class LauncherError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def read_access(setup):
    path = ROOT / ".env"
    values = {}
    allowed = {"HF_TOKEN", "HF_SPACE_ID", "SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY"}
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.is_file() else []
    for line in lines:
        if line.strip() and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            if key.strip() in allowed:
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: os.environ[key] for key in allowed if os.environ.get(key)})
    required = ["HF_TOKEN"]
    if CONTEXT_SETUPS[setup][1]:
        required += ["SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY"]
    if any(not values.get(key) for key in required):
        missing = [key for key in required if not values.get(key)]
        raise LauncherError("missing_access", "Configure .env or environment variables: " + ", ".join(missing))
    return values


def check_space(values):
    repo = values.get("HF_SPACE_ID", "tipofyzik/ielts-writing-evaluator")
    import re
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise LauncherError("invalid_space", "HF_SPACE_ID must be an owner/name repository ID.")
    base = "https://" + repo.replace("/", "-").lower() + ".hf.space"
    try:
        response = requests.get(base + "/config", headers={"Authorization": "Bearer " + values["HF_TOKEN"]}, timeout=15)
    except requests.RequestException as error:
        raise LauncherError("space_unreachable", "Cannot reach the Space. Check the connection and its HF runtime.") from error
    if response.status_code != 200:
        raise LauncherError("space_unavailable", "The Space is unavailable or access is denied. Check its runtime and HF_TOKEN.")
    try:
        found = any(d.get("api_name") == "stage5_judge" for d in response.json().get("dependencies", []))
    except (ValueError, AttributeError, TypeError) as error:
        raise LauncherError("invalid_space_response", "The Space did not return its API configuration.") from error
    if not found:
        raise LauncherError("missing_endpoint", "The running Space does not expose stage5_judge.")
    return base


def preflight(values, setup):
    _, references, aes = CONTEXT_SETUPS[setup]
    if aes:
        path = ROOT / "results/stage3/release.json"
        release = json.loads(path.read_text(encoding="utf-8"))
        bundle = path.parent / release["bundle_path"]
        if not (bundle / "manifest.json").is_file() or not (bundle / "ordinal_model.pt").is_file():
            raise LauncherError("missing_aes", "The accepted AES bundle is missing.")
    if references:
        release = json.loads((ROOT / "results/stage4/release.json").read_text(encoding="utf-8"))
        corpus = ROOT / release["corpus_path"]
        if not (corpus / "manifest.json").is_file() or not (corpus / "records.jsonl").is_file():
            raise LauncherError("missing_corpus", "The accepted corpus artifacts are missing.")
    packages = {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "huggingface-hub", "sentence-transformers", "langchain-core", "langsmith")}
    return {"status": "ready", "mode": "preflight", "context_setup": setup,
            "space": check_space(values), "packages": packages,
            "loads_models": False, "runs_inference": False,
            "scope": "Local artifact presence, installed packages and hosted endpoint availability; no DB/model inference check."}


class RemoteJudge:
    def __init__(self, values, base):
        self.registry = ModelRegistry()
        self.profile = self.registry.require_enabled("qwen25_1p5b")
        self.base = base + "/gradio_api/call/stage5_judge"
        self.headers = {"Authorization": "Bearer " + values["HF_TOKEN"]}
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.profile.repo_id, revision=self.profile.revision, local_files_only=True,
            clean_up_tokenization_spaces=False)

    def count_input_tokens(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return int(self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])

    def predict_messages(self, *, model_key, messages, deadline):
        from ielts_judge.scoring import JudgeQuotaError
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


def load_service(judge, values, setup):
    import torch
    from transformers.utils import logging
    logging.set_verbosity_error()
    from ielts_judge.scoring_runtime import load_scoring_service
    torch.set_num_threads(4)
    return load_scoring_service(judge, aes_release=ROOT / "results/stage3/release.json",
        corpus_release=ROOT / "results/stage4/release.json", project_root=ROOT,
        supabase_url=values.get("SUPABASE_URL"), supabase_key=values.get("SUPABASE_PUBLISHABLE_KEY"), setup=setup)


