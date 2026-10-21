"""Approved judge profiles; no model loading or application startup on import."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import math
import re

import yaml


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "resources" / "model_registry.yaml"
OUTPUT_SCHEMA_VERSION = "scores-v1"


@dataclass(frozen=True)
class ModelProfile:
    key: str
    label: str
    repo_id: str
    revision: str
    enabled: bool
    context_budget: int
    max_new_tokens: int
    generation_config: Mapping[str, Any]
    output_schema_version: str


class ModelRegistry:
    """Read and validate one immutable server-side allowlist."""

    def __init__(self, path: str | Path = REGISTRY_PATH) -> None:
        with Path(path).open(encoding="utf-8") as source:
            resource = yaml.safe_load(source)
        models = resource.get("models") if isinstance(resource, dict) else None
        if not isinstance(models, dict) or not models:
            raise ValueError("Registry must define a nonempty models mapping.")
        profiles = {}
        for key, entry in models.items():
            if not isinstance(key, str) or not re.fullmatch(r"[a-z0-9_]+", key):
                raise ValueError("Model keys must contain lowercase letters, digits or underscores.")
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid model profile: {key}.")
            for field in ("label", "repo_id", "revision"):
                if not isinstance(entry.get(field), str) or not entry[field].strip():
                    raise ValueError(f"{key}.{field} must be a nonempty string.")
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", entry["repo_id"]):
                raise ValueError(f"{key}.repo_id must be an HF repository ID.")
            if not re.fullmatch(r"[0-9a-f]{40}", entry["revision"]):
                raise ValueError(f"{key}.revision must be a pinned commit SHA.")
            if type(entry.get("enabled")) is not bool:
                raise ValueError(f"{key}.enabled must be a boolean.")
            for field in ("context_budget", "max_new_tokens"):
                if type(entry.get(field)) is not int or entry[field] < 1:
                    raise ValueError(f"{key}.{field} must be a positive integer.")
            if entry["max_new_tokens"] >= entry["context_budget"]:
                raise ValueError(f"{key}: output must leave room for input tokens.")
            if entry.get("output_schema_version") != OUTPUT_SCHEMA_VERSION:
                raise ValueError(f"{key}: unsupported output schema version.")
            settings = entry.get("generation_config")
            if not isinstance(settings, dict) or type(settings.get("do_sample")) is not bool:
                raise ValueError(f"{key}: generation_config needs boolean do_sample.")
            if settings.keys() - {"do_sample", "temperature", "top_p", "top_k"}:
                raise ValueError(f"{key}: unsupported generation setting.")
            for field in ("temperature", "top_p"):
                if field in settings:
                    value = settings[field]
                    if (type(value) not in (int, float) or not math.isfinite(value)
                            or value <= 0 or (field == "top_p" and value > 1)):
                        raise ValueError(f"{key}: invalid {field}.")
            if "top_k" in settings and (type(settings["top_k"]) is not int or settings["top_k"] < 0):
                raise ValueError(f"{key}: invalid top_k.")
            profiles[key] = ModelProfile(
                key=key, label=entry["label"], repo_id=entry["repo_id"],
                revision=entry["revision"], enabled=entry["enabled"],
                context_budget=entry["context_budget"], max_new_tokens=entry["max_new_tokens"],
                generation_config=MappingProxyType(dict(settings)),
                output_schema_version=entry["output_schema_version"],
            )
        self._profiles = MappingProxyType(profiles)

    def require_enabled(self, model_key: str) -> ModelProfile:
        """Reject unknown or disabled keys before any generation or download."""
        if not isinstance(model_key, str) or model_key not in self._profiles:
            raise ValueError(f"Unknown judge model key: {model_key!r}.")
        profile = self._profiles[model_key]
        if not profile.enabled:
            raise ValueError(f"Judge model is disabled: {model_key}.")
        return profile

    def enabled_profiles(self) -> tuple[ModelProfile, ...]:
        """Return the profiles available to a future model dropdown."""
        return tuple(profile for profile in self._profiles.values() if profile.enabled)
