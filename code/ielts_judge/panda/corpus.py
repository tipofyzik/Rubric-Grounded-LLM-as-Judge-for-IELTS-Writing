"""Build and verify portable, full-train PANDA corpora before SQL ingestion."""

import hashlib
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import canonical, digest, insights, parse_json, positive_int, read_json, revision, scores, sha256, write_json
from .generation import load_result, run_manifest
from ..criteria import PUBLIC_CRITERION_IDS
from ..data.preparation import file_hash, text_hash
from ..data.experiment import validate_reference_corpus

EMBEDDING_DEFAULTS = {"model_id": "BAAI/bge-base-en-v1.5",
                      "revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a", "max_length": 512}


def embedding_profile(config=None):
    if config is not None and (not isinstance(config, dict) or config.keys() - EMBEDDING_DEFAULTS.keys()):
        raise ValueError("Unknown embedding options")
    result = {**EMBEDDING_DEFAULTS, **(config or {})}
    if result["model_id"] != EMBEDDING_DEFAULTS["model_id"]:
        raise ValueError("This corpus schema supports BAAI/bge-base-en-v1.5; another encoder needs a new schema")
    revision(result["revision"])
    if not 1 <= positive_int(result["max_length"], "max_length") <= 512:
        raise ValueError("BGE max_length must not exceed 512")
    return {**result, "dimensions": 768, "input": "essay", "query_instruction": "", "document_instruction": "",
            "pooling": "model-defined", "normalize": True, "dtype": "float32", "truncation": "right",
            "distance": "cosine"}


def vector(value):
    if not isinstance(value, (list, np.ndarray)) or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in value):
        raise ValueError("Embedding must be a numeric vector")
    try:
        result = np.asarray(value, dtype=np.float32)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid embedding values") from error
    if result.shape != (768,) or not np.isfinite(result).all() or not np.isclose(np.linalg.norm(result), 1, atol=1e-4):
        raise ValueError("Embedding must have 768 finite, unit-normalized dimensions")
    return result


class Encoder:
    """One encoder/profile for documents and queries, with explicit lengths and right truncation."""

    def __init__(self, profile, *, device="cpu"):
        from sentence_transformers import SentenceTransformer

        self.profile = profile
        self.model = SentenceTransformer(profile["model_id"], revision=profile["revision"], device=device)
        self.model.max_seq_length = profile["max_length"]
        self.model.tokenizer.truncation_side = "right"
        if self.model.get_sentence_embedding_dimension() != profile["dimensions"]:
            raise ValueError("Encoder dimension differs from corpus schema")

    def encode(self, texts):
        lengths = [len(ids) for ids in self.model.tokenizer(texts, truncation=False, padding=False)["input_ids"]]
        values = self.model.encode(texts, batch_size=32, prompt="", normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(values, dtype=np.float32), lengths


def validate_record(record, schema_version="panda-corpus-v1"):
    required = {"essay_id", "essay_hash", "task_prompt", "essay", "dataset_source", "split",
                "source_scores_json", "criteria", "embedding", "original_tokens", "used_tokens", "truncated"}
    if schema_version == "panda-corpus-v2":
        required |= {"insights", "generation_status", "generation_error"}
    elif schema_version != "panda-corpus-v1":
        raise ValueError("Unsupported corpus schema")
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("Invalid corpus record fields")
    for name in ("essay_id", "task_prompt", "essay"):
        if not isinstance(record[name], str) or not record[name].strip():
            raise ValueError(f"Reference needs nonempty {name}")
    if record["split"] != "train" or record["dataset_source"] not in ("chillies", "writing9"):
        raise ValueError("Only accepted source train records may enter the corpus")
    if text_hash(record["essay"]) != record["essay_hash"]:
        raise ValueError("Reference essay hash mismatch")
    if not isinstance(record["source_scores_json"], str):
        raise ValueError("source_scores_json must preserve a source JSON string")
    criteria = record["criteria"]
    if not isinstance(criteria, dict) or set(criteria) != set(PUBLIC_CRITERION_IDS):
        raise ValueError("Four reference criteria are required")
    labels = scores({name: item["score"] for name, item in criteria.items()})
    explanation = {}
    for name, item in criteria.items():
        fields = {"score", "score_source"}
        if schema_version == "panda-corpus-v1":
            fields |= {"evidence", "why_not_higher", "why_not_lower"}
        if set(item) != fields or item["score_source"] != "source_dataset_label":
            raise ValueError("Invalid reference criterion provenance/fields")
        if schema_version == "panda-corpus-v1":
            explanation[name] = {field: item[field] for field in ("evidence", "why_not_higher", "why_not_lower")}
    if schema_version == "panda-corpus-v1":
        insights(explanation, record["essay"], labels)
    else:
        if not isinstance(record["insights"], str):
            raise ValueError("Raw insights must preserve the completion string")
        if record["generation_status"] not in ("accepted", "failed"):
            raise ValueError("Unknown source generation status")
        error = record["generation_error"]
        if (record["generation_status"] == "accepted" and error is not None) or (
            record["generation_status"] == "failed" and (not isinstance(error, str) or not error.strip())
        ):
            raise ValueError("Invalid source generation error")
    vector(record["embedding"])
    original, used = (positive_int(record[name], name) for name in ("original_tokens", "used_tokens"))
    if used > original or type(record["truncated"]) is not bool or record["truncated"] != (original > used):
        raise ValueError("Invalid embedding input length metadata")
    return record


def build_corpus(train, experiment, generation_directory, output, profile, encoder, *, batch_size=32):
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


class Corpus:
    """Validate the complete artifact on open; stream records instead of retaining all vectors in RAM."""

    def __init__(self, path):
        self.path = Path(path)
        self.manifest = manifest = read_json(self.path / "manifest.json")
        if manifest.get("schema_version") not in ("panda-corpus-v1", "panda-corpus-v2"):
            raise ValueError("Unsupported corpus schema")
        expected = {"schema_version", "dataset_manifest_sha256", "dataset_source", "train", "generator", "embedding",
                    "records_sha256", "records_digest", "corpus_id", "count", "embedding_profile_sha256", "truncated_count"}
        if set(manifest) != expected:
            raise ValueError("Invalid corpus manifest fields")
        identity = {key: value for key, value in manifest.items() if key not in
                    {"corpus_id", "count", "embedding_profile_sha256", "truncated_count"}}
        if manifest["corpus_id"] != sha256(canonical(identity)):
            raise ValueError("Corpus identity mismatch")
        profile = manifest["embedding"]
        if profile != embedding_profile({key: profile[key] for key in EMBEDDING_DEFAULTS}):
            raise ValueError("Unsupported embedding profile")
        if manifest["embedding_profile_sha256"] != sha256(canonical(profile)):
            raise ValueError("Embedding profile hash mismatch")
        digest(manifest["dataset_manifest_sha256"])
        positive_int(manifest["count"], "corpus count")
        if sha256(canonical(manifest["generator"]["rubric"])) != manifest["generator"]["rubric_sha256"]:
            raise ValueError("Generator rubric hash mismatch")
        if manifest["records_sha256"] != file_hash(self.path / "records.jsonl"):
            raise ValueError("Corpus records file hash mismatch")
        count, truncated, previous = 0, 0, None
        seen_hashes = set()
        source_rows = []
        aggregate = hashlib.sha256()
        for line, record in self.records():
            validate_record(record, manifest["schema_version"])
            if previous is not None and record["essay_id"] <= previous:
                raise ValueError("Corpus IDs must be unique and sorted")
            if record["essay_hash"] in seen_hashes:
                raise ValueError("Duplicate corpus essay hash")
            seen_hashes.add(record["essay_hash"])
            previous = record["essay_id"]
            count += 1
            truncated += int(record["truncated"])
            if record["used_tokens"] != min(record["original_tokens"], profile["max_length"]):
                raise ValueError("Embedding lengths differ from profile")
            aggregate.update((record["essay_id"] + ":" + sha256(line) + "\n").encode("utf-8"))
            source_rows.append({key: record[key] for key in ("essay_id", "essay_hash", "task_prompt", "essay", "split")})
            source_rows[-1].update(status="accepted", **{name: record["criteria"][name]["score"] for name in PUBLIC_CRITERION_IDS})
        if not count or count != manifest["count"] or count != manifest["train"]["count"] or truncated != manifest["truncated_count"]:
            raise ValueError("Corpus count/truncation metadata mismatch")
        if manifest["train"].get("policy") != "all_train" or aggregate.hexdigest() != manifest["records_digest"]:
            raise ValueError("Corpus membership digest mismatch")
        source = pd.DataFrame(source_rows)
        source.attrs["manifest_sha256"] = manifest["dataset_manifest_sha256"]
        validate_reference_corpus(source, {"schema_version": "ielts-experiment-v1", "train": manifest["train"],
            "reference_corpus": manifest["train"], "dataset_manifest_sha256": manifest["dataset_manifest_sha256"]})

    def records(self):
        with (self.path / "records.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                text = line.rstrip("\n")
                value = parse_json(text)
                if canonical(value) != text:
                    raise ValueError("Corpus records must use canonical JSON")
                yield text, value
