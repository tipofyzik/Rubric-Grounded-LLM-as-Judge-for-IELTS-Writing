"""Offline preparation of the two local IELTS sources, without model imports."""
from typing import Any, Iterator

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

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "ielts-two-sources-v2"
SPLITS = ("train", "test")
SCORE_FIELDS = dict(zip(PUBLIC_CRITERION_IDS, ("band_task", "band_coherence", "band_lexical", "band_grammar")))
TASK1_PATTERN = re.compile(r"\bwrite\s+(?:an?\s+)?(?:letter|e-?mail)\b|\bsummari[sz]e\s+the\s+information\b|\b(?:charts?|graphs?|diagrams?|tables?|maps?|plans?)\s+(?:below|above)\b", re.IGNORECASE)
COLUMN_TYPES: dict[str, Literal["string", "Int64", "Float64", "list"]] = {
    "essay_id": "string", "dataset_source": "string", "source_id": "string",
    "source_file": "string", "source_line": "Int64", "source_split": "string",
    "task_prompt": "string", "essay": "string", "evaluation": "string",
    "task_type": "string", "question_type": "string", "essay_hash": "string",
    "question_hash": "string", "word_count": "Int64", "source_scores_json": "string",
    **{name: "Int64" for name in PUBLIC_CRITERION_IDS}, "overall": "Float64",
    "overall_label": "string", "status": "string", "split": "string",
    "duplicate_of": "string", "issues": "list", "warnings": "list",
}


def file_hash(path: Any) -> str:
    """Calculate a file's SHA-256 digest for artifact and deployment verification.

    Args:
        path (Any): Filesystem location to read, validate or write.

    Returns:
        str: The computed result for the supplied inputs.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(text: str) -> Any:
    """Text hash.

    Args:
        text (str): Text used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    value = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def valid_score(value: object) -> TypeGuard[int | float]:
    """Valid score.

    Args:
        value (object): Value used by this operation.

    Returns:
        TypeGuard[int | float]: The computed result for the supplied inputs.
    """
    return (type(value) is int or type(value) is float) and math.isfinite(value) and 0 <= value <= 9


def rounded_overall(values: Any) -> Any:
    """Rounded overall.

    Args:
        values (Any): Caller-supplied settings or values used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    values = tuple(values)
    if len(values) != 4 or not all(valid_score(current_value) for current_value in values):
        return None
    mean = sum((Decimal(str(current_value)) for current_value in values), Decimal(0)) / 4
    return float((mean * 2).quantize(Decimal("1"), rounding=ROUND_HALF_UP) / 2)


def typed_frame(rows: Any) -> Any:
    """Typed frame.

    Args:
        rows (Any): Rows used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    frame = pd.DataFrame(rows, columns=list(COLUMN_TYPES))
    for column, dtype in COLUMN_TYPES.items():
        if dtype != "list":
            frame[column] = frame[column].astype(dtype)
    return frame


def source_paths(root: Any) -> dict[str, Any]:
    """Source paths.

    Args:
        root (Any): Project root used to resolve relative configuration paths.

    Returns:
        dict[str, Any]: Named result fields and metadata produced by this operation.
    """
    root = Path(root).resolve()
    return {
        "chillies": {current_text: root / "chillies - IELTS-writing-task-2-evaluation" / f"hugging_face_{current_text}.csv" for current_text in SPLITS},
        "writing9": {current_text: root / "ndtran0101 - writing9-ielts-essays/data" / f"{current_text}.jsonl" for current_text in SPLITS},
    }




def _nonstandard_number(value: Any) -> None:
    """Nonstandard number.

    Args:
        value (Any): Value used by this operation.

    Returns:
        None: This operation returns no value.
    """
    raise ValueError(f"Non-standard JSON number: {value}")


def _records(path: Any, source: Any, limit: Any) -> Iterator[Any]:
    """Records.

    Args:
        path (Any): Filesystem location to read, validate or write.
        source (Any): Source used by this operation.
        limit (Any): Limit used by this operation.

    Returns:
        Iterator[Any]: The computed result for the supplied inputs.
    """
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if source == "chillies":
            reader = csv.DictReader(stream)
            missing = {"prompt", "essay", "evaluation", "band"} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            yield from enumerate(reader, 1)
        else:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, parse_constant=_nonstandard_number)
                    if not isinstance(row, dict):
                        raise ValueError("Expected a JSON object")
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                yield line_number, row


def _criterion(value: Any, name: Any, issues: Any, warnings: Any, source: Any) -> Any:
    """Criterion.

    Args:
        value (Any): Value used by this operation.
        name (Any): Name used by this operation.
        issues (Any): Issues used by this operation.
        warnings (Any): Warnings used by this operation.
        source (Any): Source used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if value is None:
        issues.append(f"missing_score:{name}")
    elif not valid_score(value):
        issues.append(f"invalid_score:{name}")
    elif value != int(value) and source == "writing9":
        issues.append(f"fractional_criterion:{name}")
    else:
        if value != int(value):
            warnings.append(f"truncated_criterion:{name}")
        return int(value)
    return None


def extract_sources(paths: Any, extract_score: Any, limit_per_split: Any=None) -> Any:
    """Read the four explicit files; preserve rows before duplicate resolution.

    Args:
        paths (Any): Paths used by this operation.
        extract_score (Any): Extract score used by this operation.
        limit_per_split (Any): Limit per split used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if limit_per_split is not None and (type(limit_per_split) is not int or limit_per_split < 1):
        raise ValueError("limit_per_split must be a positive integer or None")
    extracted, fingerprints = {}, []
    legacy_names = [(name, IELTS_CRITERIA[name]["source_name"]) for name in CRITERION_IDS]
    for source, files in paths.items():
        rows = []
        for split in SPLITS:
            path = Path(files[split]).resolve()
            LOGGER.info("Reading %s %s", source, split)
            count, complete = 0, True
            for line, raw in _records(path, source, limit_per_split):
                if limit_per_split is not None and count == limit_per_split:
                    complete = False
                    break
                count += 1
                issues, warnings = [], []
                old = source == "chillies"
                source_id = str(line) if old else raw.get("id")
                if not isinstance(source_id, str) or not source_id.strip():
                    source_id = None
                    issues.append("missing_source_id")
                task, essay = raw.get("prompt" if old else "question"), raw.get("essay")
                for field, value in (("task_prompt", task), ("essay", essay)):
                    if not isinstance(value, str) or not value.strip():
                        issues.append(f"missing_text:{field}")
                task = task if isinstance(task, str) else None
                essay = essay if isinstance(essay, str) else None
                evaluation = raw.get("evaluation") if old else None
                if old:
                    if not isinstance(evaluation, str):
                        issues.append("missing_evaluation")
                    source_scores = {}
                    for current_index, (name, title) in enumerate(legacy_names):
                        following = legacy_names[current_index + 1][1] if current_index + 1 < len(legacy_names) else "Overall Band"
                        source_scores[LEGACY_TO_PUBLIC[name]] = extract_score(evaluation, title, following) if isinstance(evaluation, str) else None
                else:
                    source_scores = {name: raw.get(field) for name, field in SCORE_FIELDS.items()}
                    if raw.get("task_type") != "essay":
                        issues.append("unsupported_task_type")
                    if task and TASK1_PATTERN.search(task):
                        issues.append("task1_prompt")
                scores = {name: _criterion(value, name, issues, warnings, source) for name, value in source_scores.items()}
                if not old and any(value == 0 for value in scores.values()):
                    issues.append("zero_band_requires_review")
                source_overall = raw.get("band" if old else "band_overall")
                overall_label = str(source_overall).strip() if source_overall is not None else None
                if old:
                    try:
                        overall = float(overall_label) if overall_label is not None else None
                    except (ValueError, TypeError):
                        overall = None
                    if overall is not None and not valid_score(overall):
                        overall = None
                    if overall is None and overall_label != "<4":
                        issues.append("invalid_score:overall")
                else:
                    overall = float(source_overall) if valid_score(source_overall) else None
                    if overall is None:
                        issues.append("invalid_score:overall")
                    elif overall * 2 != int(overall * 2):
                        issues.append("invalid_overall_step")
                        overall = None
                    calculated = rounded_overall(source_scores.values())
                    if calculated is None:
                        issues.append("overall_comparison_unavailable")
                    elif overall is not None and overall != calculated:
                        issues.append("overall_mismatch")
                word_count = len(essay.split()) if essay is not None else None
                if word_count is not None and word_count <= 20:
                    warnings.append("response_at_most_20_whitespace_words")
                rows.append({
                    "essay_id": f"{source}:{split}:{line}", "dataset_source": source,
                    "source_id": source_id, "source_file": str(path), "source_line": line,
                    "source_split": split, "task_prompt": task, "essay": essay,
                    "evaluation": evaluation, "task_type": "essay" if old else raw.get("task_type"),
                    "question_type": raw.get("question_type"), "essay_hash": text_hash(essay),
                    "question_hash": text_hash(task), "word_count": word_count,
                    "source_scores_json": json.dumps({**source_scores, "overall": source_overall}, ensure_ascii=False, allow_nan=False),
                    **scores, "overall": overall, "overall_label": overall_label,
                    "status": "excluded" if issues else "accepted", "split": None if issues else split,
                    "duplicate_of": None, "issues": issues, "warnings": warnings,
                })
            fingerprints.append({"dataset_source": source, "split": split, "file": str(path),
                                 "sha256": file_hash(path), "rows_read": count, "complete": complete})
            LOGGER.info("Extracted %s %s: %s rows", source, split, count)
        extracted[source] = typed_frame(rows)
    return extracted, fingerprints


def _exclude(frame: Any, index: int, reason: Any) -> None:
    """Exclude.

    Args:
        frame (Any): Frame used by this operation.
        index (int): Index used by this operation.
        reason (Any): Reason used by this operation.

    Returns:
        None: This operation returns no value.
    """
    frame.at[index, "issues"] = [*frame.at[index, "issues"], reason]
    frame.at[index, "status"] = "excluded"
    frame.at[index, "split"] = None


def _signature(value: Any) -> Any:
    """Signature.

    Args:
        value (Any): Value used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    scores = json.loads(value)
    return tuple(("number", float(scores[name]))
                 if type(scores[name]) in (int, float) and math.isfinite(scores[name])
                 else ("other", json.dumps(scores[name], sort_keys=True))
                 for name in PUBLIC_CRITERION_IDS)


def resolve_duplicates(extracted: Any) -> Any:
    """Select Chillies low bands, then resolve conflicts within the selected sources.

    Args:
        extracted (Any): Extracted used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    frame = pd.concat(extracted.values(), ignore_index=True).copy(deep=True)
    # Lists are replaced rather than mutated, preserving the extracted layer.
    scores = frame[list(PUBLIC_CRITERION_IDS)]
    low_criteria = scores.notna().all(axis=1) & scores.ge(0).all(axis=1) & scores.le(3).all(axis=1)
    low_overall = frame['overall_label'].eq('<4') | frame['overall'].lt(4)
    outside_scope = frame['dataset_source'].eq('chillies') & ~(low_criteria & low_overall).fillna(False)
    for index in frame.index[outside_scope]:
        _exclude(frame, index, 'chillies_outside_low_band_scope')
    for source, group in frame.loc[frame["source_id"].notna()].groupby("dataset_source"):
        repeated = group.loc[group["source_id"].duplicated(keep=False)]
        if source == "chillies":
            continue  # Row numbers are local to each CSV, not natural source IDs.
        for _, copies in repeated.groupby("source_id"):
            if len(copies[["essay_hash", "question_hash"]].drop_duplicates()) > 1:
                for index in copies.index:
                    _exclude(frame, index, "source_id_collision")
    scoped = frame.loc[~outside_scope]
    repeated = scoped["essay_hash"].notna() & scoped["essay_hash"].duplicated(keep=False)
    for _, group in scoped.loc[repeated].groupby("essay_hash", sort=False):
        if len({_signature(value) for value in group["source_scores_json"]}) > 1:
            for index in group.index:
                _exclude(frame, index, "duplicate_essay_conflict")
            continue
        candidates = group.loc[group["status"].eq("accepted")]
        if len(candidates) < 2:
            continue
        ordered = candidates.assign(_priority=candidates["source_split"].map({"test": 0, "train": 1})).sort_values(
            ["_priority", "dataset_source", "source_id", "source_line"])
        if group["question_hash"].nunique() > 1:
            for index in ordered.index:
                frame.at[index, "warnings"] = [*frame.at[index, "warnings"], "duplicate_essay_different_tasks"]
        keeper = ordered.iloc[0]["essay_id"]
        for index in ordered.index[1:]:
            frame.at[index, "status"] = "duplicate"
            frame.at[index, "split"] = None
            frame.at[index, "duplicate_of"] = keeper
            frame.at[index, "issues"] = [*frame.at[index, "issues"], "duplicate_essay"]
    return frame


@dataclass
class PreparedDataset:
    """Store the prepared dataset values shared by this component's callers."""
    rows: pd.DataFrame
    sources: list
    extracted: dict
    parent_manifest: str | None = None

    def split(self, name: Any) -> Any:
        """Split.

        Args:
            name (Any): Name used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        if name not in SPLITS:
            raise ValueError(f"Unknown split: {name}")
        return self.rows.loc[self.rows["status"].eq("accepted") & self.rows["split"].eq(name)].copy()


def validate_splits(train: Any, test: Any) -> None:
    """Reject missing labels, duplicate IDs and essay leakage at the loader boundary.

    Args:
        train (Any): Train used by this operation.
        test (Any): Test used by this operation.

    Returns:
        None: This operation returns no value.
    """
    for name, frame in (("train", train), ("test", test)):
        if not frame["status"].eq("accepted").all() or not frame["split"].eq(name).all():
            raise ValueError(f"Invalid statuses/split in {name}")
        values = frame[list(PUBLIC_CRITERION_IDS)]
        if not (values.notna().all().all() and values.ge(0).all().all() and values.le(9).all().all() and values.mod(1).eq(0).all().all()):
            raise ValueError(f"Invalid criterion labels in {name}")
    combined = pd.concat([train, test], ignore_index=True)
    if combined["essay_id"].isna().any() or combined["essay_id"].duplicated().any():
        raise ValueError("Accepted essay IDs must be unique")
    if combined["essay_hash"].isna().any() or combined["essay_hash"].duplicated().any():
        raise ValueError("Duplicate essays or train/test leakage")




def load_prepared(output: Any, source: str="combined", *, allow_partial: bool=False) -> Any:
    """Validate a prepared dataset and load its train and test splits.

    Args:
        output (Any): Existing dataset directory containing manifest.json and the splits.
        source (str): Keep both sources (combined), or filter to chillies or writing9.
        allow_partial (bool): Allow an explicitly partial release, such as a smoke-test dataset.

    Returns:
        Any: (train DataFrame, test DataFrame, manifest dict). Both frames carry
            the dataset version and manifest identity in their attrs.

    Raises:
        ValueError: The source or release is unsupported, partial data is disallowed,
            or the manifest, files or splits fail validation.
    """
    output = Path(output).resolve()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in ("ielts-two-sources-v1", SCHEMA_VERSION):
        raise ValueError("Unsupported prepared dataset schema")
    if manifest.get("partial_input") and not allow_partial:
        raise ValueError("Partial dataset release is not allowed for runtime")
    if source not in ("combined", "chillies", "writing9"):
        raise ValueError(f"Unknown dataset source: {source}")
    required = {"train.parquet", "test.parquet", "audit.json", "score_profile_counts.json"}
    if manifest.get("save_intermediate_datasets"):
        required.update(("excluded.parquet", "not_selected.parquet",
                         "extracted/chillies.parquet", "extracted/writing9.parquet"))
    if manifest.get("selection") is not None:
        required.update(("selection.json", "score_profiles_before_selection.json"))
    if manifest.get("profile_ids_file"):
        required.add("score_profile_ids.json")
    if not required.issubset(manifest.get("artifacts", {})) or manifest.get("criteria") != list(PUBLIC_CRITERION_IDS):
        raise ValueError("Incomplete dataset manifest or wrong criterion order")
    for name, info in manifest["artifacts"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output):
            raise ValueError("Artifact path escapes dataset directory")
        if file_hash(path) != info["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {name}")
    train, test = (pd.read_parquet(output / f"{split}.parquet") for split in SPLITS)
    validate_splits(train, test)
    if source != "combined":
        train, test = (frame.loc[frame["dataset_source"].eq(source)].copy() for frame in (train, test))
    for frame in (train, test):
        frame.attrs.update(dataset_version=manifest["schema_version"], manifest_sha256=file_hash(manifest_path))
    return train, test, manifest


def legacy_labels(frame: Any) -> Any:
    """One explicit adapter to legacy AES criterion order; no dataset mutation.

    Args:
        frame (Any): Frame used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    return frame[list(PUBLIC_CRITERION_IDS)].rename(columns={current_value: current_key for current_key, current_value in LEGACY_TO_PUBLIC.items()})[list(CRITERION_IDS)]
