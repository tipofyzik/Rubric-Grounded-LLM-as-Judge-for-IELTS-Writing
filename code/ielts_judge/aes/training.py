"""Full prepared-train AES fitting and ID-aligned evaluation exports."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from ..criteria import PUBLIC_CRITERION_IDS
from ..data.experiment import validate_reference_corpus
from ..data.preparation import file_hash
from .inference import write_json
from .OrdinalRegression import ENCODER_ID, ENCODER_REVISION, OrdinalRegression, positive_integer, token_lengths


def aes_options(config=None):
    defaults = {"encoder_name": ENCODER_ID, "encoder_revision": ENCODER_REVISION,
                "max_length": 512, "learning_rate": 2e-5, "seed": 42,
                "epochs": 3, "batch_size": 8, "bundle_path": None}
    if config is not None and (not isinstance(config, dict) or config.keys() - defaults.keys()):
        raise ValueError("Unknown AES options")
    options = {**defaults, **(config or {})}
    for key in ("epochs", "batch_size", "max_length"):
        positive_integer(options[key], key)
    return options


def metrics_report(truth, predictions, train_labels):
    """Baseline constants are chosen from train; test is used only for metrics."""
    result = {}
    baseline = np.floor(np.median(train_labels, axis=0)).astype(int)
    for index, name in enumerate(PUBLIC_CRITERION_IDS):
        actual, predicted = truth[:, index], predictions[:, index]
        # QWK is undefined only when both marginals put all mass on one category.
        qwk = (None if len(np.unique(np.concatenate([actual, predicted]))) < 2 else
               float(cohen_kappa_score(actual, predicted, labels=list(range(10)), weights="quadratic")))
        result[name] = {
            "count": len(actual), "mae": float(mean_absolute_error(actual, predicted)),
            "qwk": qwk, "qwk_undefined_reason": "single identical category" if qwk is None else None,
            "train_median_baseline": int(baseline[index]),
            "baseline_mae": float(mean_absolute_error(actual, np.full(len(actual), baseline[index]))),
        }
    return result


def run_aes(train, test, experiment, result_root, *, config=None, train_model=True):
    """One path for main.py and the Kaggle CLI; never resplit or sample train."""
    validate_reference_corpus(train, experiment)
    options = aes_options(config)
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset_manifest_sha256": experiment["dataset_manifest_sha256"],
        "dataset_source": experiment["dataset_source"], "train": experiment["train"],
        "validation": "none; epochs and settings fixed before fitting",
        "test_usage": "metrics only; adjustable evaluation pool, not untouched final holdout",
        "encoder_initialization": "pretrained pinned HF revision",
        "labels": "source dataset labels; not certified examiner marks",
    }
    fit_seconds = None
    if train_model:
        bundle = result_root / "aes_bundle"
        if bundle.exists():
            raise FileExistsError(f"Use a new run directory; AES bundle exists: {bundle}")
        model = OrdinalRegression(**{key: options[key] for key in (
            "encoder_name", "encoder_revision", "max_length", "learning_rate", "seed")})
        labels = train[list(PUBLIC_CRITERION_IDS)]
        encoded = model.ordinal_encode_multi(labels, {i: i for i in range(10)}, 10)
        started = time.perf_counter()
        model.fit(train.essay.tolist(), encoded, epochs=options["epochs"], batch_size=options["batch_size"])
        fit_seconds = time.perf_counter() - started
        model.save_bundle(bundle, provenance=provenance)
        # Evaluate the persisted artifact, not only the still-live training object.
        device = model.device
        del model
        model = OrdinalRegression.load_bundle(bundle, device=device)
    else:
        if not options["bundle_path"]:
            raise ValueError("AES inference requires aes.bundle_path")
        bundle = Path(options["bundle_path"])
        model = OrdinalRegression.load_bundle(bundle)
        if model.bundle_manifest is None:
            raise ValueError("AES loader did not return bundle metadata")
        saved = model.bundle_manifest["provenance"]
        if any(saved.get(key) != provenance[key] for key in ("dataset_manifest_sha256", "dataset_source", "train")):
            raise ValueError("AES bundle training provenance does not match this experiment train")
    started = time.perf_counter()
    predictions = model.predict(test.essay.tolist(), batch_size=options["batch_size"])
    inference_seconds = time.perf_counter() - started
    table = pd.DataFrame(predictions, columns=[f"pred_{name}" for name in PUBLIC_CRITERION_IDS], dtype="int64")
    table.insert(0, "essay_id", test.essay_id.to_numpy())
    table["dataset_manifest"] = experiment["dataset_manifest_sha256"]
    table["aes_bundle_manifest_sha256"] = file_hash(bundle / "manifest.json")
    lengths = token_lengths(model.tokenizer, test.essay.tolist())
    table["original_tokens"] = lengths
    table["used_tokens"] = [min(n, model.max_length) for n in lengths]
    table["truncated"] = [n > model.max_length for n in lengths]
    table.to_csv(result_root / "aes_predictions.csv", index=False)
    report = {
        "bundle_path": str(bundle), "bundle_manifest_sha256": file_hash(bundle / "manifest.json"),
        "train_count": len(train), "test_count": len(test), "valid_count": len(predictions),
        "fit_seconds": fit_seconds, "inference_seconds": inference_seconds,
        "test_truncated_count": sum(n > model.max_length for n in lengths),
        "criterion_semantics": "essay_only; first score is not task-aware TR",
        "metrics": metrics_report(test[list(PUBLIC_CRITERION_IDS)].to_numpy(dtype=int),
                                  np.asarray(predictions), train[list(PUBLIC_CRITERION_IDS)].to_numpy(dtype=int)),
    }
    write_json(result_root / "aes_report.json", report)
    return report
