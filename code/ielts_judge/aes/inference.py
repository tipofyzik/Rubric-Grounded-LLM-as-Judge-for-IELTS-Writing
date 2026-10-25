"""Complete AES bundles: local reconstruction, integrity and semantic checks."""

import json
import os
from pathlib import Path
import tempfile
from typing import cast

import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from ..criteria import PUBLIC_CRITERION_IDS
from ..data.preparation import file_hash
from .OrdinalRegression import OrdinalRegression

SCHEMA = "ielts-aes-bundle-v1"
ARCHITECTURE = {"name": "masked_mean_four_ordinal_heads_v1", "num_classes": 10,
                "num_tasks": 4, "head_dropout": 0.2}
DECODING = {"name": "count_sigmoid_above_threshold", "probability_threshold": 0.5,
            "comparison": "strictly_greater"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_bundle(model, save_dir, *, provenance=None):
    """Publish a new directory only after every bundle component was written."""
    destination = Path(save_dir)
    if destination.exists():
        raise FileExistsError(f"AES bundle already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aes-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        torch.save(model.model.state_dict(), stage / "ordinal_model.pt")
        model.model.encoder.config.save_pretrained(stage / "encoder")
        model.tokenizer.save_pretrained(stage / "tokenizer")
        artifacts = {p.relative_to(stage).as_posix(): file_hash(p)
                     for p in sorted(stage.rglob("*")) if p.is_file()}
        manifest = {
            "schema_version": SCHEMA, "architecture": ARCHITECTURE,
            "criteria": list(PUBLIC_CRITERION_IDS), "bands": list(range(10)),
            "criterion_semantics": {PUBLIC_CRITERION_IDS[0]: "essay_only_legacy_proxy_not_task_aware"},
            "decoding": DECODING,
            "input": {"mode": "essay_only", "max_length": model.max_length,
                      "special_tokens_included": True, "truncation_side": "right",
                      "training_overlength": "truncate", "inference_overlength": "truncate_with_metadata"},
            "encoder": {"id": model.encoder_name, "revision": model.encoder_revision},
            "training": dict(model.training_metadata), "provenance": provenance or {},
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "artifacts": artifacts,
        }
        write_json(stage / "manifest.json", manifest)
        os.replace(stage, destination)
    model.bundle_manifest = manifest
    return manifest


def read_manifest(save_dir):
    root = Path(save_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Complete AES bundle manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("AES manifest must be a JSON object")
    if (manifest.get("schema_version") != SCHEMA or manifest.get("architecture") != ARCHITECTURE
            or manifest.get("criteria") != list(PUBLIC_CRITERION_IDS)
            or manifest.get("bands") != list(range(10)) or manifest.get("decoding") != DECODING
            or manifest.get("criterion_semantics") != {
                PUBLIC_CRITERION_IDS[0]: "essay_only_legacy_proxy_not_task_aware"}):
        raise ValueError("Unsupported AES bundle architecture, criteria, bands or semantics")
    policy = manifest.get("input", {})
    if not isinstance(policy, dict):
        raise ValueError("Unsupported AES input policy")
    length = policy.get("max_length")
    expected = {"mode": "essay_only", "max_length": length, "special_tokens_included": True,
                "truncation_side": "right", "training_overlength": "truncate",
                "inference_overlength": "truncate_with_metadata"}
    if policy != expected or type(length) is not int or length < 3:
        raise ValueError("Unsupported AES input policy")
    training = manifest.get("training", {})
    if (not isinstance(training, dict) or training.get("status") != "trained" or
            type(training.get("train_count")) is not int or training["train_count"] < 1):
        raise ValueError("AES bundle has no completed training")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not {"ordinal_model.pt", "encoder/config.json",
                                              "tokenizer/tokenizer_config.json"}.issubset(artifacts):
        raise ValueError("Incomplete AES artifact inventory")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {"manifest.json"}
    if actual != set(artifacts):
        raise ValueError("AES artifact inventory does not match the directory")
    for name, digest in artifacts.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != digest:
            raise ValueError(f"AES artifact integrity failure: {name}")
    return manifest


def load_bundle(save_dir, *, device=None):
    """No pretrained downloads or partial state loading are permitted here."""
    manifest = read_manifest(save_dir)
    root = Path(save_dir)
    config = AutoConfig.from_pretrained(root / "encoder", local_files_only=True)
    encoder = cast(PreTrainedModel, AutoModel.from_config(config))
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    if tokenizer.truncation_side != "right":
        raise ValueError("AES requires right truncation")
    model = OrdinalRegression(
        encoder=encoder, tokenizer=tokenizer, device=device,
        encoder_name=manifest["encoder"]["id"], encoder_revision=manifest["encoder"]["revision"],
        max_length=manifest["input"]["max_length"],
        learning_rate=manifest["training"]["learning_rate"], seed=manifest["training"]["seed"],
    )
    state = torch.load(root / "ordinal_model.pt", map_location="cpu", weights_only=True)
    model.model.load_state_dict(state, strict=True)
    model.model.eval()
    model.training_metadata = manifest["training"]
    model.loss_history = manifest["training"]["loss_history"]
    model.bundle_manifest = manifest
    return model
