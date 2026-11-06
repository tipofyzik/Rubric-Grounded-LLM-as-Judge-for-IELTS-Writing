"""Explicit experiment and report options; plotting never changes inference."""
from typing import Any

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from ielts_judge.ModelRegistry import ModelRegistry
from ielts_judge.Prompts import CONTEXT_SETUPS
from offline_research.evaluation.Metrics import METRICS

PLOT_TYPES = (
    "metric_comparison", "metric_bars", "metric_heatmap", "calibration", "calibration_by_prediction", "bias",
    "error_histogram", "error_histogram_std", "error_ridge", "confusion_matrix",
    "score_distribution", "coverage", "prompt_comparison", "repeatability", "latency",
)
DEFAULT_REPORT = {
    "metrics": ["within_one_accuracy", "mae", "rmse", "pearson", "spearman", "qwk"],
    "plots": ["metric_comparison", "calibration"],
    "formats": ["png"], "dpi": 200, "layout": "grid", "style": {},
}


def fingerprint(value: Any) -> Any:
    """Fingerprint.

    Args:
        value (Any): Value used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def report_options(options: Any=None) -> Any:
    """Report options.

    Args:
        options (Any): Options used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    options = {} if options is None else options
    if not isinstance(options, dict) or options.keys() - DEFAULT_REPORT.keys():
        raise ValueError("Unknown report option; smoothing is not supported")
    result = {**deepcopy(DEFAULT_REPORT), **deepcopy(options)}
    if isinstance(result["plots"], list):
        result["plots"] = ["prompt_comparison" if candidate_path == "ablation" else candidate_path for candidate_path in result["plots"]]
    if isinstance(result["plots"], list) and "score_response" in result["plots"]:
        result["plots"] = list(dict.fromkeys("calibration" if candidate_path == "score_response" else candidate_path for candidate_path in result["plots"]))
    for key, allowed in (("metrics", METRICS), ("plots", PLOT_TYPES), ("formats", ("png", "svg", "webp"))):
        value = result[key]
        if (not isinstance(value, list) or len(value) != len(set(value))
                or any(item not in allowed for item in value)):
            raise ValueError(f"Invalid report.{key}; allowed: {', '.join(allowed)}")
    if not result["metrics"] or not result["formats"]:
        raise ValueError("Report needs metrics and output formats")
    if type(result["dpi"]) is not int or not 30 <= result["dpi"] <= 600:
        raise ValueError("report.dpi must be an integer from 30 to 600")
    if result["layout"] not in ("grid", "single"):
        raise ValueError("report.layout must be grid or single")
    if not isinstance(result["style"], dict) or result["style"].keys() - {"aes", "models", "setups", "rc"}:
        raise ValueError("Style supports aes, models, setups and rc Matplotlib settings")
    return result


def load_config(path: Any, root: Any) -> Any:
    """Load config.

    Args:
        path (Any): Filesystem location to read, validate or write.
        root (Any): Project root used to resolve relative configuration paths.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "experiment_name", "inputs", "models", "model_registry", "setups",
                "evaluation", "runtime", "report"}
    if not isinstance(config, dict) or set(config) != required or config["schema_version"] != "model-evaluation-v1":
        raise ValueError("Expected model-evaluation-v1 configuration with explicit sections")
    if not isinstance(config["experiment_name"], str) or not config["experiment_name"].strip():
        raise ValueError("experiment_name must be nonempty")
    if set(config["inputs"]) != {"dataset", "aes_release", "corpus_release"}:
        raise ValueError("inputs must name dataset, aes_release and corpus_release")
    for name in (*config["inputs"].values(), config["model_registry"]):
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Input paths must be relative to the project snapshot")
    for key, allowed in (("setups", CONTEXT_SETUPS), ("models", None)):
        items = config[key]
        if not isinstance(items, list) or not items or len(set(items)) != len(items):
            raise ValueError(f"{key} must be a nonempty list of unique names")
        if allowed is not None and set(items) - set(allowed):
            raise ValueError("Unknown context setup")
    registry = ModelRegistry(Path(root) / config["model_registry"])
    for key in config["models"]:
        registry.require_enabled(key)
    if set(config["evaluation"]) != {"sample_size", "seed", "repeats"}:
        raise ValueError("evaluation needs sample_size, seed and repeats")
    from ielts_judge.data.Experiment import evaluation_options
    evaluation_options({current_key: current_value for current_key, current_value in config["evaluation"].items() if current_key != "repeats"})
    runtime = config["runtime"]
    if set(runtime) != {"judge_device", "context_device", "aes_batch_size", "retrieval_k", "timeout_seconds", "max_input_chars"}:
        raise ValueError("Unknown or missing runtime option")
    if runtime["judge_device"] not in ("cpu", "cuda") or runtime["context_device"] not in ("cpu", "cuda"):
        raise ValueError("Devices must be cpu or cuda")
    for value in (config["evaluation"]["repeats"], runtime["aes_batch_size"], runtime["timeout_seconds"], runtime["max_input_chars"]):
        if type(value) is not int or value < 1:
            raise ValueError("Repeats, batch size and limits must be positive integers")
    if type(runtime["retrieval_k"]) is not int or not 1 <= runtime["retrieval_k"] <= 20:
        raise ValueError("retrieval_k must be from 1 to 20")
    config["report"] = report_options(config["report"])
    return config
