"""Full-train, resumable PANDA generation with one atomic result per essay ID."""
from typing import Any

from pathlib import Path


from ielts_judge.panda.Artifacts import canonical, insights, parse_completion, parse_json, parse_legacy_completion, positive_int, read_json, revision, sha256
from ielts_judge.Criteria import IELTS_CRITERIA, PUBLIC_CRITERION_IDS
from ielts_judge.data.Experiment import validate_reference_corpus

PROMPT_VERSIONS = {"simple_json": "panda-insights-json-v2", "legacy_text": "panda-insights-text-v1",
                   "original_text": "panda-insights-original-085e4a1-v1",
                   "adjacent_text": "panda-insights-adjacent-text-v5"}
GENERATOR_DEFAULTS = {
    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "max_input_tokens": 4096, "max_new_tokens": 1024, "seed": 42, "prompt_variant": "simple_json",
    "batch_size": 12,
}


def generation_profile(config: Any=None) -> dict[str, Any]:
    """Generation profile.

    Args:
        config (Any): Config used by this operation.

    Returns:
        dict[str, Any]: Named result fields and metadata produced by this operation.
    """
    if config is not None and (not isinstance(config, dict) or config.keys() - GENERATOR_DEFAULTS.keys()):
        raise ValueError("Unknown insight generation options")
    result = {**GENERATOR_DEFAULTS, **(config or {})}
    revision(result["revision"])
    if not isinstance(result["model_id"], str) or not result["model_id"].strip():
        raise ValueError("Generator model_id is required")
    for name in ("max_input_tokens", "max_new_tokens", "batch_size"):
        positive_int(result[name], name)
    if type(result["seed"]) is not int or result["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if result["prompt_variant"] not in PROMPT_VERSIONS:
        raise ValueError("Unknown prompt_variant")
    return {**result, "prompt_version": PROMPT_VERSIONS[result["prompt_variant"]],
            "validation_version": "panda-insights-validation-v2",
            "rubric_sha256": sha256(canonical(IELTS_CRITERIA)),
            "rubric": parse_json(canonical(IELTS_CRITERIA)),
            "rubric_source": "project abbreviated IELTS rubric; not verbatim official descriptors",
            "do_sample": False, "dtype": "float16", "input_policy": "full task+essay; reject over budget"}


def result_path(output: Any, essay_id: str | None) -> Any:
    """Result path.

    Args:
        output (Any): Output used by this operation.
        essay_id (str | None): Stable source essay identifier used for reference exclusion.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    return Path(output) / "responses" / (sha256(essay_id) + ".json")


def row_input(row: Any) -> dict[str, Any]:
    """Row input.

    Args:
        row (Any): Row used by this operation.

    Returns:
        dict[str, Any]: Named result fields and metadata produced by this operation.
    """
    return {"essay_id": row["essay_id"], "essay_hash": row["essay_hash"],
            "task_prompt": row["task_prompt"], "essay": row["essay"],
            "scores": {name: int(row[name]) for name in PUBLIC_CRITERION_IDS}}


def run_manifest(train: Any, experiment: Any, profile: Any) -> dict[str, Any]:
    """Run manifest.

    Args:
        train (Any): Train used by this operation.
        experiment (Any): Experiment used by this operation.
        profile (Any): Profile used by this operation.

    Returns:
        dict[str, Any]: Named result fields and metadata produced by this operation.
    """
    validate_reference_corpus(train, experiment)
    return {"schema_version": "panda-generation-v1", "dataset_manifest_sha256": experiment["dataset_manifest_sha256"],
            "dataset_source": experiment["dataset_source"], "train": experiment["train"], "generator": profile}


def parse_insights(completion: Any, record: Any, profile: Any, *, allow_partial: bool=False) -> Any:
    """Validate insights by default; allow_partial returns unvalidated data for failed records only.

    Args:
        completion (Any): Completion used by this operation.
        record (Any): Record used by this operation.
        profile (Any): Profile used by this operation.
        allow_partial (bool): Allow partial used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    legacy = profile.get("prompt_variant") in ("legacy_text", "original_text", "adjacent_text")
    parsed = parse_legacy_completion(completion, record["scores"],
        explicit_bands=profile.get("prompt_variant") == "adjacent_text", allow_partial=allow_partial) if legacy else parse_completion(completion)
    if allow_partial:
        return (parsed or None) if legacy else parsed
    return insights(parsed, record["essay"], record["scores"], require_quote=not legacy)


def load_result(output: Any, row: Any, manifest: Any) -> Any:
    """Load result.

    Args:
        output (Any): Output used by this operation.
        row (Any): Row used by this operation.
        manifest (Any): Manifest used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    path = result_path(output, row["essay_id"])
    if not path.exists():
        return None
    result = read_json(path)
    if result.get("input_sha256") != sha256(canonical(row_input(row))) or result.get("run_sha256") != sha256(canonical(manifest)):
        raise ValueError(f"Stale insight result: {row['essay_id']}")
    if result.get("status") not in ("accepted", "failed"):
        raise ValueError("Unknown insight result status")
    if result["status"] == "accepted":
        if parse_insights(result["completion"], row_input(row), manifest["generator"]) != result["insights"] or result["diagnostics"].get("stop_reason") != "eos":
            raise ValueError("Accepted insight differs from the saved completed response")
    return result






