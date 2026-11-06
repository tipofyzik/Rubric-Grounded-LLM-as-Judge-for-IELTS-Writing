"""Offline corpus workflow; shared runtime owns read contracts."""
from typing import Any

import hashlib
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd
from ielts_judge.panda.Artifacts import canonical, digest, insights, parse_json, positive_int, read_json, revision, scores, sha256, write_json
from offline_research.panda.Contracts import load_result, run_manifest
from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from ielts_judge.data.Preparation import file_hash, text_hash
from ielts_judge.data.Experiment import validate_reference_corpus
from ielts_judge.panda.Corpus import Corpus, validate_record, vector

def build_corpus(train: Any, experiment: Any, generation_directory: Any, output: Any, profile: Any, encoder: Any, *, batch_size: int=32) -> Any:
    """Build corpus.

    Args:
        train (Any): Train used by this operation.
        experiment (Any): Experiment used by this operation.
        generation_directory (Any): Generation directory used by this operation.
        output (Any): Output used by this operation.
        profile (Any): Profile used by this operation.
        encoder (Any): Encoder used by this operation.
        batch_size (int): Batch size used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    positive_int(batch_size, "batch_size")
    generation_directory, output = Path(generation_directory), Path(output)
    generation = read_json(generation_directory / "generation_manifest.json")
    if (generation_directory / "comparison_manifest.json").exists():
        raise ValueError("Comparison selection cannot be used as a full-train corpus")
    if generation != run_manifest(train, experiment, generation["generator"]):
        raise ValueError("Generation release/train differs from the experiment")
    rows = train.sort_values("essay_id").to_dict("records")
    # Require every source response, including failures; preserve completions verbatim.
    for row in rows:
        result = load_result(generation_directory, row, generation)
        if result is None or not isinstance(result.get("completion"), str):
            raise ValueError(f"Full-train insights are incomplete: {row['essay_id']}")
    if getattr(encoder, "profile", None) != profile:
        raise ValueError("Encoder profile mismatch")
    output.mkdir(parents=True, exist_ok=False)
    records_digest = hashlib.sha256()
    truncated_count = 0
    with (output / "records.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            embeddings, lengths = encoder.encode([row["essay"] for row in batch])
            if len(embeddings) != len(batch) or len(lengths) != len(batch):
                raise ValueError("Encoder outputs do not align with essay IDs")
            for row, embedding, length in zip(batch, embeddings, lengths):
                result = load_result(generation_directory, row, generation)
                if result is None:
                    raise ValueError("Insight disappeared during corpus construction")
                used = min(length, profile["max_length"])
                record = {name: row[name] for name in ("essay_id", "essay_hash", "task_prompt", "essay", "dataset_source", "split")}
                record.update(source_scores_json=str(row.get("source_scores_json", "{}")),
                              criteria={name: {"score": int(row[name]), "score_source": "source_dataset_label"}
                                        for name in PUBLIC_CRITERION_IDS},
                              insights=result["completion"], generation_status=result["status"],
                              generation_error=result.get("error"),
                              embedding=vector(embedding).tolist(), original_tokens=int(length),
                              used_tokens=int(used), truncated=length > used)
                validate_record(record, "panda-corpus-v2")
                line = canonical(record)
                stream.write(line + "\n")
                records_digest.update((row["essay_id"] + ":" + sha256(line) + "\n").encode("utf-8"))
                truncated_count += int(record["truncated"])
    identity = {"schema_version": "panda-corpus-v2", "dataset_manifest_sha256": generation["dataset_manifest_sha256"],
                "dataset_source": generation["dataset_source"], "train": generation["train"],
                "generator": generation["generator"], "embedding": profile,
                "records_sha256": file_hash(output / "records.jsonl"), "records_digest": records_digest.hexdigest()}
    manifest = {**identity, "corpus_id": sha256(canonical(identity)), "count": len(rows),
                "embedding_profile_sha256": sha256(canonical(profile)), "truncated_count": truncated_count}
    write_json(output / "manifest.json", manifest)
    Corpus(output)
    return manifest

