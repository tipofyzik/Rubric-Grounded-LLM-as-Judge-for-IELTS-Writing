"""Full-train reference membership and reproducible selections from source test."""

import hashlib
import json
import re

import pandas as pd

from ..criteria import PUBLIC_CRITERION_IDS
from .preparation import text_hash, validate_splits

SCHEMA_VERSION = "ielts-experiment-v1"


def evaluation_options(config=None):
    if config is None:
        config = {}
    if not isinstance(config, dict) or config.keys() - {"sample_size", "seed"}:
        raise ValueError("Unknown evaluation options; expected sample_size and seed")
    options = {"sample_size": None, "seed": 42, **config}
    size, seed = options["sample_size"], options["seed"]
    if size is not None and (type(size) is not int or size < 1):
        raise ValueError("evaluation.sample_size must be a positive integer or null")
    if type(seed) is not int or seed < 0:
        raise ValueError("evaluation.seed must be a nonnegative integer")
    return options


def _validate_texts(frame):
    for name in ("essay_id", "task_prompt", "essay"):
        if not frame[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Experiment rows require nonempty {name}")
    if not frame.essay.map(text_hash).eq(frame.essay_hash).all():
        raise ValueError("Essay text and essay_hash do not match")


def _records_hash(frame):
    """Include content and labels, so the same IDs cannot hide changed records."""
    digest = hashlib.sha256()
    columns = ["essay_id", "task_prompt", "essay", "essay_hash", *PUBLIC_CRITERION_IDS]
    for row in frame.sort_values("essay_id")[columns].itertuples(index=False, name=None):
        record = [*row[:4], *map(int, row[4:])]
        digest.update(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_reference_corpus(frame, experiment_manifest):
    """Enforce the full train described by a prepared experiment, before embedding."""
    if experiment_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported experiment manifest schema")
    corpus = experiment_manifest["reference_corpus"]
    if corpus.get("policy") != "all_train":
        raise ValueError("Reference corpus policy must be all_train")
    if any(corpus[key] != experiment_manifest["train"][key] for key in ("count", "records_sha256")):
        raise ValueError("Reference corpus descriptor must match the entire train")
    if frame.attrs.get("manifest_sha256") != experiment_manifest["dataset_manifest_sha256"]:
        raise ValueError("Reference corpus belongs to a different dataset release")
    validate_splits(frame, frame.iloc[:0])
    _validate_texts(frame)
    if len(frame) != corpus["count"] or _records_hash(frame) != corpus["records_sha256"]:
        raise ValueError("Reference corpus must match the entire experiment train")


def prepare_experiment(train, test_pool, *, config=None, source="combined", saved_manifest=None):
    """Return selected test and its passport; never change train or source splits.

    A saved manifest takes precedence over sampling config and is fully checked
    against the current release, memberships, content and deterministic ordering.
    """
    if source not in ("combined", "chillies", "writing9"):
        raise ValueError("Unknown experiment dataset source")
    validate_splits(train, test_pool)
    _validate_texts(train)
    _validate_texts(test_pool)
    dataset_hash = train.attrs.get("manifest_sha256")
    if (not isinstance(dataset_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", dataset_hash)
            or test_pool.attrs.get("manifest_sha256") != dataset_hash):
        raise ValueError("Train and test require the same verified dataset manifest")
    if saved_manifest is not None:
        if not isinstance(saved_manifest, dict) or saved_manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported experiment manifest schema")
        config = saved_manifest.get("evaluation", {}).get("config")
    options = evaluation_options(config)
    size = options["sample_size"]
    if size is not None and size > len(test_pool):
        raise ValueError("evaluation.sample_size exceeds the available source test")
    ordered = sorted(test_pool.essay_id, key=lambda essay_id: (
        hashlib.sha256(f"{options['seed']}:{essay_id}".encode("utf-8")).digest(), essay_id))
    ids = sorted(ordered if size is None else ordered[:size])
    selected = test_pool.set_index("essay_id", drop=False).loc[ids].reset_index(drop=True)
    selected.attrs = dict(test_pool.attrs)
    train_hash = _records_hash(train)
    train_questions = set(train.task_prompt.map(text_hash))
    pool_questions = test_pool.task_prompt.map(text_hash)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_manifest_sha256": dataset_hash,
        "dataset_source": source,
        "train": {"policy": "all_train", "count": len(train), "records_sha256": train_hash},
        "reference_corpus": {"policy": "all_train", "count": len(train), "records_sha256": train_hash,
                             "state": "membership_only; insights and embeddings are separate artifacts"},
        "test_pool": {"count": len(test_pool), "records_sha256": _records_hash(test_pool)},
        "evaluation": {"config": options, "count": len(selected), "essay_ids": ids,
                       "ordering": "SHA256(seed:essay_id), then essay_id; exported rows sorted by ID"},
        "question_policy": {
            "name": "essay_disjoint_questions_allowed",
            "shared_train_test_questions": len(train_questions.intersection(pool_questions)),
            "test_pool_rows_with_train_question": int(pool_questions.isin(train_questions).sum()),
            "selected_rows_with_train_question": int(selected.task_prompt.map(text_hash).isin(train_questions).sum()),
        },
        "protocol": {
            "demo": "AES priors + rubric + retrieved train insights -> prompt-engineered judge",
            "sft": "optional research; not required for the demo",
            "validation": "not reserved; any future training validation needs its own protocol",
            "test_usage": "adjustable evaluation pool; not certified as untouched final holdout",
            "demo_examples": "selected at UI stage; not an independent benchmark",
            "test_labels": "metrics only; never supplied to the judge or reference corpus",
            "retrieval_exclusions": ["test records", "query essay ID", "query essay hash"],
            "aes_training_provenance": "unknown",
            "label_provenance": "source dataset reference labels; not certified examiner marks",
            "criteria": list(PUBLIC_CRITERION_IDS), "criterion_range": [0, 9], "criterion_step": 1,
        },
    }
    if saved_manifest is not None and manifest != saved_manifest:
        raise ValueError("Saved experiment manifest does not match this release, source or selection")
    validate_reference_corpus(train, manifest)
    return selected, manifest
