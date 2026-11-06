"""A resumable paired prompt experiment; this selection is never a reference corpus."""
from typing import Any

from pathlib import Path

from ielts_judge.panda.Artifacts import read_json, write_json
from offline_research.panda.Contracts import GENERATOR_DEFAULTS, generation_profile, result_path, row_input
from offline_research.panda.Generation import run_insights


def run_comparison(train: Any, experiment: Any, output: Any, profile: Any, selection: Any, generate: Any, *, limit: Any=None, retry_failed: bool=False) -> Any:
    """generate(records, variant) uses one shared model and identical decoding for both prompts.

    Args:
        train (Any): Train used by this operation.
        experiment (Any): Experiment used by this operation.
        output (Any): Output used by this operation.
        profile (Any): Profile used by this operation.
        selection (Any): Selection used by this operation.
        generate (Any): Generate used by this operation.
        limit (Any): Limit used by this operation.
        retry_failed (bool): Retry failed used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if (selection.get("schema_version") != "panda-comparison-selection-v1"
        or selection.get("dataset_manifest_sha256") != experiment["dataset_manifest_sha256"]
        or selection.get("train_records_sha256") != experiment["train"]["records_sha256"]):
        raise ValueError("Comparison selection belongs to another dataset/train")
    ids = selection.get("essay_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(value, str) for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Comparison needs unique train essay IDs")
    by_id = {row["essay_id"]: row for row in train.to_dict("records")}
    if not set(ids) <= by_id.keys():
        raise ValueError("Comparison IDs must belong to train")
    profiles = {variant: generation_profile({**{key: profile[key] for key in GENERATOR_DEFAULTS},
                                           "prompt_variant": variant}) for variant in ("simple_json", "legacy_text")}
    output = Path(output)
    if (output / "generation_manifest.json").exists():
        raise ValueError("Generation directory cannot be used for comparison")
    manifest = {"schema_version": "panda-prompt-comparison-v1", "generator": profile,
                "selection": selection, "profiles": profiles}
    passport = output / "comparison_manifest.json"
    if passport.exists() and read_json(passport) != manifest:
        raise ValueError("Comparison profile/selection changed; use a new output directory")
    if not passport.exists() and output.exists() and any(output.iterdir()):
        # Environment/experiment metadata may already be present, but never adopt orphan results.
        if any((output / variant).exists() for variant in profiles):
            raise ValueError("Orphan comparison responses without manifest")
    write_json(passport, manifest)
    reports = {}
    for variant, selected_profile in profiles.items():
        reports[variant] = run_insights(train, experiment, output / variant, selected_profile,
            lambda records: generate(records, variant), limit=limit, retry_failed=retry_failed, essay_ids=ids)
    paired = []
    for essay_id in ids:
        record = row_input(by_id[essay_id])
        record["variants"] = {}
        for variant in profiles:
            path = result_path(output / variant, essay_id)
            record["variants"][variant] = read_json(path) if path.exists() else {"status": "pending"}
        paired.append(record)
    write_json(output / "paired_review.json", {
        "scope": "Same selected essays; automatic acceptance measures format/contract, not semantic quality.",
        "validation": {"simple_json": "Exact essay quote and three fields per criterion; boundary placeholders only.",
                       "legacy_text": "Original three-bullet format; prose evidence allowed; boundary placeholders only. Comparison only."},
        "review_questions": ["Does evidence support the claim and belong to this criterion?",
            "Does why_not_higher name a specific limitation?",
            "Does why_not_lower name a demonstrated strength rather than another weakness?",
            "Is the reasoning consistent with the essay/task, including conflicts with source labels?"],
        "records": paired})
    report = {"schema_version": "panda-comparison-report-v1", "selected_count": len(ids),
              "variants": reports, "semantic_winner": None,
              "scope": "Pilot comparison only. Human review required; selection is not representative of full train."}
    write_json(output / "comparison_report.json", report)
    return report
