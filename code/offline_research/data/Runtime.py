"""Explicit adapters from verified dataset rows to model/evaluation inputs."""

import json
from pathlib import Path
from typing import Any, Sequence, TypeAlias, cast

import pandas as pd

from ielts_judge.Criteria import CRITERION_IDS, LEGACY_TO_PUBLIC
from ielts_judge.data.Preparation import legacy_labels

Submission: TypeAlias = str | dict[str, Any]
SubmissionBatch: TypeAlias = pd.Series | pd.DataFrame | Sequence[Submission]


def submission_text(value: Submission) -> str:
    """Submission text.

    Args:
        value (Submission): Value used by this operation.

    Returns:
        str: The computed result for the supplied inputs.
    """
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise TypeError("Expected an essay string or a submission record")
    task, essay = value.get("task_prompt"), value.get("essay")
    if not isinstance(task, str) or not task.strip() or not isinstance(essay, str) or not essay.strip():
        raise ValueError("Submissions require a nonempty task_prompt and essay")
    return json.dumps({"task_prompt": task, "essay": essay}, ensure_ascii=False)


def submission_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    # The projection fixes the dictionary keys to these four string columns.
    """Submission records.

    Args:
        frame (pd.DataFrame): Frame used by this operation.

    Returns:
        list[dict[str, Any]]: The computed result for the supplied inputs.
    """
    records = cast(list[dict[str, Any]], frame[["essay_id", "task_prompt", "essay", "essay_hash"]].to_dict("records"))
    for row in records:
        row["dataset_manifest"] = frame.attrs.get("manifest_sha256")
    return records


def evaluation_samples(values: SubmissionBatch) -> list[tuple[str | int, Submission, str | None]]:
    """Evaluation samples.

    Args:
        values (SubmissionBatch): Caller-supplied settings or values used by this operation.

    Returns:
        list[tuple[str | int, Submission, str | None]]: The computed result for the supplied inputs.
    """
    if isinstance(values, pd.DataFrame):
        return [(record["essay_id"], record, record["dataset_manifest"]) for record in submission_records(values)]
    return [(value["essay_id"], value, value.get("dataset_manifest")) if isinstance(value, dict)
            else (current_index, value, None) for current_index, value in enumerate(values)]


def prediction_record(sample_id: Any, submission: Any, manifest: Any, answer: Any) -> Any:
    """Prediction record.

    Args:
        sample_id (Any): Sample id used by this operation.
        submission (Any): Submission used by this operation.
        manifest (Any): Manifest used by this operation.
        answer (Any): Answer used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if not isinstance(submission, dict):
        return answer
    return {"sample_id": sample_id, "essay_id": sample_id, "dataset_manifest": manifest, "answer": answer}


def legacy_prediction(scores: Any) -> Any:
    """Legacy prediction.

    Args:
        scores (Any): Four criterion scores using the public IELTS score contract.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if not isinstance(scores, dict):
        return dict.fromkeys(CRITERION_IDS, float("nan"))
    return {name: scores.get(name, scores.get(LEGACY_TO_PUBLIC[name], float("nan"))) for name in CRITERION_IDS}


def export_reference(frame: Any, path: Any) -> Any:
    """Export reference.

    Args:
        frame (Any): Frame used by this operation.
        path (Any): Filesystem location to read, validate or write.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    reference = legacy_labels(frame).copy()
    reference.insert(0, "index", frame["essay_id"].to_numpy())
    reference["dataset_manifest"] = frame.attrs.get("manifest_sha256")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(path, index=False)
    return path


def convert_to_llm_dataset(frame: Any, *, test_data: bool=False) -> Any:
    """Convert to llm dataset.

    Args:
        frame (Any): Frame used by this operation.
        test_data (bool): Test data used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    labels = legacy_labels(frame).to_dict("records")
    rows = []
    for record, scores in zip(submission_records(frame), labels):
        answer = json.dumps(scores, ensure_ascii=False)
        prompt = ("Evaluate the following task and essay. Return ONLY JSON with four criterion scores.\n"
                  + submission_text(record) + "\n### Answer:\n" + ("" if test_data else answer))
        rows.append({"essay_id": record["essay_id"], "dataset_manifest": record["dataset_manifest"],
                     "text": prompt, "evaluation": answer if test_data else None})
    return pd.DataFrame(rows)
