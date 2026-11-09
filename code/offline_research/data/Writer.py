"""Offline writer workflow; shared runtime owns read contracts."""
from typing import Any

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import unicodedata
from typing import Literal, TypeGuard
import pandas as pd
from ielts_judge.Criteria import CRITERION_IDS, IELTS_CRITERIA, LEGACY_TO_PUBLIC, PUBLIC_CRITERION_IDS
from ielts_judge.data.Preparation import COLUMN_TYPES, LOGGER, SCHEMA_VERSION, SCORE_FIELDS, SPLITS, file_hash, validate_splits

def validate_output(output: Any, paths: Any) -> Any:
    """Check the raw source paths and keep release writes outside their directories.

    Args:
        output (Any): Proposed release directory.
        paths (Any): Raw file paths grouped by source and split.

    Returns:
        Any: Resolved Path for the release directory.

    Raises:
        FileNotFoundError: A raw source file is missing.
        NotADirectoryError: The output path names an existing file.
        ValueError: The output overlaps a raw source directory.
    """
    output = Path(output).resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    for files in paths.values():
        for path in files.values():
            path = Path(path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if output == path.parent or output.is_relative_to(path.parent) or path.is_relative_to(output):
                raise ValueError("Output must be outside source folders and must not contain source files")
    return output


def save_prepared(prepared: Any, output: Any, paths: Any, *, selection: Any=None, save_intermediate_datasets: bool=False) -> Any:
    """Write the selected splits and publish their manifest last.

    When rebuilding, invalidate the existing manifest before replacing files
    so a failed write cannot leave an old manifest attached to new content.

    Args:
        prepared (Any): PreparedDataset with cleaned rows and source provenance.
        output (Any): Directory for the prepared release.
        paths (Any): Raw source paths grouped by source and split.
        selection (Any): Optional score-profile and row selection settings.
        save_intermediate_datasets (bool): Also save extracted, excluded and unselected rows.

    Returns:
        Any: Resolved Path to the prepared release.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from offline_research.data.Analysis import PreparedDataAnalyzer
    from offline_research.data.Selection import select_dataset, selection_options
    output = validate_output(output, paths)
    if type(save_intermediate_datasets) is not bool:
        raise ValueError("dataset.save_intermediate_datasets must be a boolean")
    options = selection_options(selection)
    validate_splits(prepared.split("train"), prepared.split("test"))
    rows = prepared.rows
    reports = {}
    if options is not None:
        before_analyzer = PreparedDataAnalyzer(rows)
        before_ids = before_analyzer.profiles(with_ids=True)
        rows, selection_report = select_dataset(rows, before_ids, options)
        reports.update({"score_profiles_before_selection.json": before_analyzer.profiles(),
                        "selection.json": selection_report})
    train, test = (rows.loc[rows.status.eq("accepted") & rows.split.eq(split)].copy() for split in SPLITS)
    validate_splits(train, test)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    arrow_types = {"string": pa.string(), "Int64": pa.int64(), "Float64": pa.float64(), "list": pa.list_(pa.string())}
    schema = pa.schema([(name, arrow_types[dtype]) for name, dtype in COLUMN_TYPES.items()])
    frames = {"train.parquet": train, "test.parquet": test}
    if save_intermediate_datasets:
        frames["excluded.parquet"] = rows.loc[~rows.status.isin(("accepted", "not_selected"))]
        frames["not_selected.parquet"] = rows.loc[rows.status.eq("not_selected")]
        frames.update({f"extracted/{source}.parquet": frame for source, frame in prepared.extracted.items()})
    artifacts = {}
    for name, frame in frames.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Writing %s: %s rows", name, len(frame))
        table = pa.Table.from_pandas(frame.sort_values("essay_id"), schema=schema, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
        artifacts[name] = {"sha256": file_hash(path), "rows": len(frame)}
    analyzer = PreparedDataAnalyzer(rows)
    reports.update({"audit.json": analyzer.summary(), "score_profile_counts.json": analyzer.profiles(),
                    "score_profile_ids.json": analyzer.profiles(with_ids=True)})
    for name, report in reports.items():
        path = output / name
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        artifacts[name] = {"sha256": file_hash(path)}
    for path in analyzer.plot_score_distributions(output / "plots"):
        artifacts[path.relative_to(output).as_posix()] = {"sha256": file_hash(path)}
    from offline_research.data import DatasetTransformer, Analysis as analysis, Profiles as profiles, Selection as selection
    from ielts_judge.data import Preparation as contracts
    code_files = [Path(contracts.__file__), Path(__file__), *[Path(module.__file__) for module in (DatasetTransformer, analysis, profiles, selection)]]
    manifest = {
        "schema_version": SCHEMA_VERSION, "sources": prepared.sources,
        "preparation_mode": "extracted_cache" if prepared.parent_manifest else "original_sources",
        "parent_manifest_sha256": prepared.parent_manifest,
        "source_field_mapping": {"writing9": SCORE_FIELDS, "chillies": {
            LEGACY_TO_PUBLIC[name]: IELTS_CRITERIA[name]["source_name"] for name in CRITERION_IDS}},
        "partial_input": any(not item["complete"] for item in prepared.sources),
        "criteria": list(PUBLIC_CRITERION_IDS), "criterion_range": [0, 9], "criterion_step": 1,
        "score_policies": {
            "chillies": "validate extracted source criteria in 0-9, truncate to integer labels; preserve source_scores_json and evaluation; never impute labels from overall",
            "writing9": "finite integer criteria 0-9, exclude zero; source overall must equal four-source-criterion mean rounded to half bands, ties up; never rewrite source overall",
        },
        "source_selection_policy": {
            "chillies": "all four integer labels after source-score truncation must be 0-3 and source overall must be <4 (numeric or literal <4); apply before duplicate/conflict resolution",
            "writing9": "retain existing source validation and within-source duplicate/conflict rules; no additional score-range filter",
            "out_of_scope": "out-of-scope Chillies cannot exclude or replace Writing9; exclusions are counted in audit.json",
        },
        "overall_policy": "retain numeric source overall unchanged; Chillies <4 retained in overall_label with null numeric overall; plots truncate numeric labels only",
        "normalization": "NFKC/LF/outer-strip for hashes only; model texts unchanged",
        "split_policy": "preserve source splits; within selected sources duplicate essays prefer test; exclude every selected copy of conflicting raw criterion labels",
        "profile_policy": reports["score_profile_counts.json"]["type_assignment"],
        "profile_ids_file": "score_profile_ids.json",
        "selection": options,
        "selection_policy": ("profile quotas after cleanup; original splits preserved; unused rows counted in audit.json"
                             if options is not None else "disabled; preserve all accepted rows"),
        "save_intermediate_datasets": save_intermediate_datasets,
        "artifact_policy": ("train/test, intermediate Parquet datasets and reports" if save_intermediate_datasets
                            else "train/test Parquet and reports only; intermediate datasets remain in memory"),
        "code_sha256": {candidate_path.name: file_hash(candidate_path) for candidate_path in code_files},
        "versions": {"pandas": pd.__version__, "pyarrow": pa.__version__}, "artifacts": artifacts,
    }
    if options is not None:
        import scipy
        manifest["versions"]["scipy"] = scipy.__version__
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return output

