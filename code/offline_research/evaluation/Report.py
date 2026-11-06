"""ID-aligned numeric tables and reproducible, model-only evaluation reports."""
from typing import Any

from pathlib import Path
import json
from datetime import datetime, timezone
from importlib.metadata import version
import platform

import numpy as np
import pandas as pd

from ielts_judge.Criteria import LEGACY_TO_PUBLIC, PUBLIC_CRITERION_IDS
from ielts_judge.data.Preparation import file_hash
from ielts_judge.Scores import public_scores
from offline_research.evaluation.Config import fingerprint, report_options
from offline_research.evaluation.Metrics import METRICS, integer_bands, metric_values

CRITERIA = tuple(PUBLIC_CRITERION_IDS)
SHORT_NAMES = dict(zip(CRITERIA, ("TR", "CC", "LR", "GRA")))
AES_KEY = "aes"


def reference_frame(frame: Any) -> Any:
    """Reference frame.

    Args:
        frame (Any): Frame used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    frame = frame.rename(columns={"index": "essay_id", **{current_key: current_value for current_key, current_value in LEGACY_TO_PUBLIC.items() if current_key != current_value}}).copy()
    if "essay_id" not in frame or not set(CRITERIA).issubset(frame):
        raise ValueError("Reference needs essay_id and four public criterion labels")
    if frame.essay_id.isna().any() or frame.essay_id.duplicated().any():
        raise ValueError("Reference essay IDs must be nonempty and unique")
    frame["essay_id"] = frame.essay_id.astype(str)
    if frame.essay_id.str.strip().eq("").any() or frame.essay_id.duplicated().any():
        raise ValueError("Reference essay IDs must be nonempty")
    frame[list(CRITERIA)] = integer_bands(frame[list(CRITERIA)].to_numpy())
    return frame


def aligned_results(records: Any, truth: Any, plan: Any) -> Any:
    """Aligned results.

    Args:
        records (Any): Records used by this operation.
        truth (Any): Truth used by this operation.
        plan (Any): Plan used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    truth = reference_frame(truth)
    ids = truth.essay_id.tolist()
    if set(ids) != set(plan["essay_ids"]) or len(ids) != len(plan["essay_ids"]):
        raise ValueError("Report reference differs from planned essay IDs")
    manifest = plan.get("dataset_manifest_sha256")
    if manifest and "dataset_manifest" in truth and set(truth.dataset_manifest) != {manifest}:
        raise ValueError("Reference belongs to a different dataset release")
    groups = ([(AES_KEY, "aes", 0)] if plan.get("include_aes", True) else []) + [(model, setup, repeat)
              for model in plan["models"] for setup in plan["setups"] for repeat in range(plan["repeats"])]
    allowed = set(groups)
    indexed = {}
    for record in records:
        key = (record["model_key"], record["setup"], record["repeat"], str(record["essay_id"]))
        if key in indexed:
            raise ValueError(f"Duplicate prediction key: {key}")
        if key[:3] not in allowed or key[3] not in ids:
            raise ValueError(f"Unplanned prediction key: {key}")
        if manifest and record.get("dataset_manifest") != manifest:
            raise ValueError("Prediction belongs to a different dataset release")
        if record.get("status") not in {"ok", "error"}:
            raise ValueError("Prediction status must be ok or error")
        indexed[key] = record
    rows = []
    for model, setup, repeat in groups:
        for essay_id in ids:
            item = indexed.get((model, setup, repeat, essay_id))
            row = {"model_key": model, "setup": setup, "repeat": repeat, "essay_id": essay_id,
                   "status": "not_run", "error_code": "not_run", **dict.fromkeys(CRITERIA, np.nan)}
            if item is not None:
                row.update(status=item["status"], error_code=item.get("error", {}).get("code", ""))
                if item["status"] == "ok":
                    try:
                        row.update(public_scores(item.get("scores")))
                    except (TypeError, ValueError):
                        row.update(status="error", error_code="invalid_output")
                timings = item.get("timings", {})
                generation = item.get("metadata", {}).get("generation", {})
                for key in ("total_seconds", "judge_seconds"):
                    row[key] = timings.get(key)
                for key in ("gpu_work_seconds", "gpu_peak_allocated_mib", "input_tokens", "generated_tokens"):
                    row[key] = generation.get(key)
            rows.append(row)
    result = pd.DataFrame(rows)
    labels = truth[["essay_id", *CRITERIA]].rename(columns={current_criterion: "true_" + current_criterion for current_criterion in CRITERIA})
    result = result.merge(labels, on="essay_id", how="left", validate="many_to_one")
    return result


def report_tables(frame: Any, truth: Any, plan: Any) -> dict[str, Any]:
    """Report tables.

    Args:
        frame (Any): Frame used by this operation.
        truth (Any): Truth used by this operation.
        plan (Any): Plan used by this operation.

    Returns:
        dict[str, Any]: Named result fields and metadata produced by this operation.
    """
    groups = list(frame.groupby(["model_key", "setup", "repeat"], sort=False))
    common = set(truth.essay_id.astype(str))
    for _, group in groups:
        common.intersection_update(group.loc[group.status.eq("ok"), "essay_id"])
    metric_rows, coverage, failures, curves, calibration, errors, confusion, repeats = [], [], [], [], [], [], [], []
    for (model, setup, repeat), group in groups:
        identity = {"model_key": model, "setup": setup, "repeat": repeat}
        valid = group.loc[group.status.eq("ok")]
        coverage.append({**identity, "planned": len(group), "processed": int(group.status.ne("not_run").sum()),
                         "valid": len(valid), "coverage": len(valid) / len(group) if len(group) else None})
        for (status, code), subset in group.groupby(["status", "error_code"], dropna=False):
            failures.append({**identity, "status": status, "error_code": code, "count": len(subset)})
        for population, selected in (("own_valid", valid), ("common_valid", valid.loc[valid.essay_id.isin(common)])):
            per_criterion = {}
            for criterion in CRITERIA:
                values, reasons = metric_values(selected["true_" + criterion].to_numpy(), selected[criterion].to_numpy())
                per_criterion[criterion] = values
                for metric in METRICS:
                    metric_rows.append({**identity, "population": population, "criterion": criterion, "metric": metric,
                                        "value": values[metric], "undefined_reason": reasons.get(metric), "n_valid": len(selected)})
            for metric in METRICS:
                values = [per_criterion[current_criterion][metric] for current_criterion in CRITERIA]
                metric_rows.append({**identity, "population": population, "criterion": "macro", "metric": metric,
                                    "value": float(np.mean(values)) if all(current_value is not None for current_value in values) else None,
                                    "undefined_reason": None if all(current_value is not None for current_value in values) else "one or more criteria undefined",
                                    "n_valid": len(selected)})
        for criterion in CRITERIA:
            actual = valid["true_" + criterion].to_numpy(dtype=int)
            predicted = valid[criterion].to_numpy(dtype=int)
            difference = predicted - actual
            for band in range(10):
                mask = actual == band
                curves.append({**identity, "criterion": criterion, "band": band, "count": int(mask.sum()),
                               "mean_prediction": float(predicted[mask].mean()) if mask.any() else None,
                               "bias": float(difference[mask].mean()) if mask.any() else None})
                mask = predicted == band
                calibration.append({**identity, "criterion": criterion, "band": band, "count": int(mask.sum()),
                                    "mean_reference": float(actual[mask].mean()) if mask.any() else None})
                for predicted_band in range(10):
                    count = int(((actual == band) & (predicted == predicted_band)).sum())
                    denominator = int((actual == band).sum())
                    confusion.append({**identity, "criterion": criterion, "true_band": band, "predicted_band": predicted_band,
                                      "count": count, "row_fraction": count / denominator if denominator else None})
            for difference_value in range(-9, 10):
                count = int((difference == difference_value).sum())
                errors.append({**identity, "criterion": criterion, "error": difference_value, "count": count,
                               "fraction": count / len(valid) if len(valid) else None})
    metrics = pd.DataFrame(metric_rows)
    summary = []
    keys = ["population", "model_key", "setup", "criterion", "metric"]
    for key, group in metrics.groupby(keys, sort=False):
        values = group.value.dropna()
        summary.append({**dict(zip(keys, key)), "mean": float(values.mean()) if len(values) else None,
                        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                        "n_runs": len(group), "n_defined_runs": len(values),
                        "n_valid_min": int(group.n_valid.min()), "n_valid_max": int(group.n_valid.max()),
                        "undefined_reason": "; ".join(sorted(set(group.undefined_reason.dropna().astype(str)))) or None})
    for (model, setup, essay_id), group in frame.loc[frame.model_key.ne(AES_KEY)].groupby(["model_key", "setup", "essay_id"], sort=False):
        valid = group.loc[group.status.eq("ok")]
        for criterion in CRITERIA:
            values = valid[criterion].to_numpy()
            measured = len(values) >= 2
            repeats.append({"model_key": model, "setup": setup, "essay_id": essay_id, "criterion": criterion,
                            "planned_repeats": len(group), "valid_repeats": len(valid),
                            "range": int(values.max() - values.min()) if measured else None,
                            "all_equal": bool(np.all(values == values[0])) if measured else None})
    reference_counts = [{"criterion": current_criterion, "band": current_band, "count": int(truth[current_criterion].eq(current_band).sum())} for current_criterion in CRITERIA for current_band in range(10)]
    base = metrics.loc[metrics.population.eq("common_valid") & metrics.setup.eq("base")][["model_key", "repeat", "criterion", "metric", "value"]].rename(columns={"value": "base_value"})
    changes = metrics.loc[metrics.population.eq("common_valid") & metrics.model_key.ne(AES_KEY)].merge(
        base, on=["model_key", "repeat", "criterion", "metric"], how="left", validate="many_to_one")
    changes["delta_from_base"] = changes.value - changes.base_value
    return {
        "predictions": frame, "metrics_per_run": metrics, "metrics_summary": pd.DataFrame(summary),
        "coverage": pd.DataFrame(coverage), "failures": pd.DataFrame(failures),
        "score_response": pd.DataFrame(curves), "calibration": pd.DataFrame(calibration),
        "error_distribution": pd.DataFrame(errors), "confusion": pd.DataFrame(confusion),
        "repeatability": pd.DataFrame(repeats, columns=["model_key", "setup", "essay_id", "criterion", "planned_repeats", "valid_repeats", "range", "all_equal"]),
        "reference_distribution": pd.DataFrame(reference_counts),
        "common_ids": pd.DataFrame({"essay_id": sorted(common)}),
        "prompt_comparison": changes,
    }


def write_report(records: Any, truth: Any, plan: Any, output: Any, options: Any=None, *, model_loads: Any=()) -> Any:
    """Write report.

    Args:
        records (Any): Records used by this operation.
        truth (Any): Truth used by this operation.
        plan (Any): Plan used by this operation.
        output (Any): Output used by this operation.
        options (Any): Options used by this operation.
        model_loads (Any): Model loads used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    options = report_options(options)
    truth = reference_frame(truth)
    frame = aligned_results(records, truth, plan)
    tables = report_tables(frame, truth, plan)
    tables["model_loads"] = pd.DataFrame(model_loads, columns=["model_key", "session", "load_seconds", "requested_device", "scope", "cuda_allocated_mib"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    display = tables["metrics_summary"].copy()
    display["display_value"] = display.apply(lambda current_record: "undefined" if pd.isna(current_record["mean"]) else (
        f"{current_record['mean']:.3f} +/- {current_record['std']:.3f}" if pd.notna(current_record["std"]) else f"{current_record['mean']:.3f} (one defined run)"), axis=1)
    display.to_csv(output / "metrics_display.csv", index=False)
    from offline_research.evaluation.Plots import render_plots
    generated = render_plots(tables, plan, output / "plots", options)
    manifest = {
        "schema_version": "model-evaluation-report-v1", "service_evaluation": "deferred",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_source_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in ("Metrics.py", "Config.py", "Report.py", "Plots.py")},
        "report_environment": {"python": platform.python_version(), "packages": {name: version(name) for name in ("numpy", "pandas", "matplotlib", "scikit-learn", "scipy")}},
        "plan": plan, "report_options": options, "reference_ids_sha256": fingerprint(sorted(truth.essay_id)),
        "common_valid_count": len(tables["common_ids"]), "complete": bool(frame.status.ne("not_run").all()),
        "undefined_policy": "null with reason; no imputation or rounding",
        "comparison_population": "IDs valid in every planned model/setup/repeat, including AES",
        "error_bars": "sample STD between repeated metrics; unmeasured for one defined run",
        "aes_semantics": "essay-only; TR is a legacy proxy", "plots": generated,
    }
    manifest["files"] = {candidate_path.relative_to(output).as_posix(): file_hash(candidate_path) for candidate_path in sorted(output.rglob("*")) if candidate_path.is_file()}
    (output / "report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return output
