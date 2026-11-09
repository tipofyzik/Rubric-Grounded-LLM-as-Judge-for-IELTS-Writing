"""Prepare real AES/BGE context once, using the accepted train corpus offline."""
from typing import Any

from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import numpy as np

from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from ielts_judge.data.Preparation import file_hash, text_hash
from ielts_judge.Prompts import ReferenceContext


def asset_paths(root: Any, config: Any) -> dict[str, Any]:
    """Resolve evaluation assets inside the checkout and verify their release descriptors.

    Args:
        root (Any): Project root used to resolve paths in the evaluation configuration.
        config (Any): Evaluation config whose inputs select the dataset, AES and PANDA releases.

    Returns:
        dict[str, Any]: Absolute directory paths keyed by dataset, aes and corpus.

    Raises:
        ValueError: An asset path escapes the checkout or a release does not match.
    """
    root = Path(root).resolve()
    aes_release_path = root / config["inputs"]["aes_release"]
    corpus_release_path = root / config["inputs"]["corpus_release"]
    aes = json.loads(aes_release_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_release_path.read_text(encoding="utf-8"))
    aes_path = (aes_release_path.parent / aes["bundle_path"]).resolve()
    corpus_path = (root / corpus["corpus_path"]).resolve()
    dataset = (root / config["inputs"]["dataset"]).resolve()
    for path in (aes_path, corpus_path, dataset):
        if not path.is_relative_to(root):
            raise ValueError("Evaluation asset path escapes the project snapshot")
    if file_hash(aes_path / "manifest.json") != aes["bundle_manifest_sha256"]:
        raise ValueError("AES bundle differs from accepted release")
    corpus_manifest = json.loads((corpus_path / "manifest.json").read_text(encoding="utf-8"))
    if corpus_manifest["corpus_id"] != corpus["corpus_id"]:
        raise ValueError("Corpus differs from accepted release")
    return {"dataset": dataset, "aes": aes_path, "corpus": corpus_path}


def nearest_references(matrix: Any, records: Any, query_vector: Any, *, essay_id: str | None, essay_hash: Any, reference_count: int) -> Any:
    """Rank eligible PANDA records by cosine similarity.

    Args:
        matrix (Any): Matrix of reference embeddings in record order.
        records (Any): Corpus records aligned row-for-row with the embedding matrix.
        query_vector (Any): Query embedding in the same vector space.
        essay_id (str | None): Source ID to exclude.
        essay_hash (Any): Query content fingerprint used to exclude identical essays.
        reference_count (int): Maximum number of references to return.

    Returns:
        Any: (ReferenceContext dictionaries, aligned similarity scores), both lists
            in descending similarity order with essay ID as the tie breaker.
    """
    from ielts_judge.panda.Corpus import vector
    query_vector = vector(query_vector)
    similarities = matrix @ query_vector / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector))
    eligible = np.array([record["essay_id"] != essay_id and record["essay_hash"] != essay_hash for record in records])
    order = sorted(np.flatnonzero(eligible), key=lambda current_index: (-float(similarities[current_index]), records[current_index]["essay_id"]))[:reference_count]
    references = []
    for index in order:
        row = records[index]
        references.append(asdict(ReferenceContext(
            essay_id=row["essay_id"], task_prompt=row["task_prompt"], essay=row["essay"],
            scores={criterion_identifier: row["criteria"][criterion_identifier]["score"] for criterion_identifier in PUBLIC_CRITERION_IDS}, insights=row["insights"],
        )))
    return references, [float(similarities[current_index]) for current_index in order]


def prepare_contexts(paths: Any, frame: Any, experiment: Any, *, device: str="cpu", batch_size: int=8, reference_count: int=1) -> Any:
    """Precompute AES scores and offline PANDA references for the selected evaluation essays.

    Args:
        paths (Any): Verified AES and PANDA directories from asset_paths.
        frame (Any): Evaluation DataFrame with essay_id, essay_hash and essay columns.
        experiment (Any): Experiment manifest identifying the prepared dataset and train split.
        device (str): Device on which to load the AES model and query encoder.
        batch_size (int): Number of essays per AES prediction batch.
        reference_count (int): Maximum number of training references per essay.

    Returns:
        Any: (context records, timing summary). Records follow the input row order.
            Timings cover batch preparation across the entire selection.

    Raises:
        ValueError: Asset provenance differs from the experiment or a test essay
            overlaps with the reference corpus.
    """
    from ielts_judge.aes.OrdinalRegression import OrdinalRegression, token_lengths
    from ielts_judge.panda.Corpus import Corpus, Encoder
    import torch

    started = time.perf_counter()
    corpus = Corpus(paths["corpus"])
    manifest = corpus.manifest
    if manifest["schema_version"] != "panda-corpus-v2":
        raise ValueError("Evaluation requires the accepted raw-text corpus v2")
    if (manifest["dataset_manifest_sha256"] != experiment["dataset_manifest_sha256"]
            or manifest["train"] != experiment["train"]):
        raise ValueError("Corpus and evaluation train differ")
    source = [record for _, record in corpus.records()]
    if set(frame.essay_id).intersection(current_record["essay_id"] for current_record in source) or set(frame.essay_hash).intersection(current_record["essay_hash"] for current_record in source):
        raise ValueError("Evaluation essays must not be reference train records")
    matrix = np.array([current_record["embedding"] for current_record in source], dtype=np.float32)
    model = OrdinalRegression.load_bundle(paths["aes"], device=device)
    try:
        provenance = model.bundle_manifest["provenance"]
        if (provenance["dataset_manifest_sha256"] != experiment["dataset_manifest_sha256"]
                or provenance["train"] != experiment["train"]):
            raise ValueError("AES training provenance differs from evaluation train")
        tick = time.perf_counter()
        predictions = model.predict(frame.essay.tolist(), batch_size=batch_size)
        lengths = token_lengths(model.tokenizer, frame.essay.tolist())
        aes_seconds = time.perf_counter() - tick
        max_length = model.max_length
    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    encoder = Encoder(manifest["embedding"], device=device)
    try:
        tick = time.perf_counter()
        vectors, query_lengths = encoder.encode(frame.essay.tolist())
        encoder_seconds = time.perf_counter() - tick
    finally:
        del encoder
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    rows = []
    tick = time.perf_counter()
    for index, row in enumerate(frame.itertuples(index=False)):
        references, similarities = nearest_references(matrix, source, vectors[index],
            essay_id=row.essay_id, essay_hash=row.essay_hash, reference_count=reference_count)
        query_limit = manifest["embedding"]["max_length"]
        rows.append({
            "essay_id": row.essay_id, "essay_hash": row.essay_hash,
            "aes": {"scores": dict(zip(PUBLIC_CRITERION_IDS, predictions[index])),
                    "input": {"original_tokens": lengths[index], "used_tokens": min(lengths[index], max_length),
                              "max_length": max_length, "truncated": lengths[index] > max_length},
                    "semantics": "essay_only; task_response is a legacy proxy, not task-aware TR"},
            "references": references,
            "retrieval": {"status": "ok" if references else "retrieval_empty", "corpus_id": manifest["corpus_id"],
                          "embedding_profile_sha256": manifest["embedding_profile_sha256"], "similarities": similarities,
                          "query": {"original_tokens": query_lengths[index], "used_tokens": min(query_lengths[index], query_limit),
                                    "max_length": query_limit, "truncated": query_lengths[index] > query_limit}},
        })
    return rows, {"aes_batch_seconds": aes_seconds, "bge_batch_seconds": encoder_seconds,
                  "retrieval_batch_seconds": time.perf_counter() - tick, "total_seconds": time.perf_counter() - started,
                  "count": len(rows), "timing_scope": "whole context preparation; not individual submission latency"}


class PreparedAES:
    """Serve saved AES scores for one evaluation essay."""
    def __init__(self, context: Any) -> None:
        """Bind the adapter to one prepared context record.

        Args:
            context (Any): Record with essay_hash and the saved aes result.
        """
        self.context = context

    def predict_essay(self, essay: str) -> Any:
        """Return the saved AES result for the essay bound to this adapter.

        Args:
            essay (str): Essay text whose content fingerprint must match the saved record.

        Returns:
            Any: Saved scores and model input-length metadata.

        Raises:
            ValueError: The text belongs to a different essay.
        """
        if text_hash(essay) != self.context["essay_hash"]:
            raise ValueError("Prepared AES context belongs to another essay")
        return self.context["aes"]


class PreparedRetriever:
    """Serve saved PANDA references through the scoring service's lookup interface."""
    def __init__(self, context: Any) -> None:
        """Bind the adapter to one prepared context record.

        Args:
            context (Any): Record with essay ID, content fingerprint, references and retrieval metadata.
        """
        self.context = context

    def retrieve(self, essay: str, *, essay_id: str | None=None, timeout: float | None=None) -> dict[str, Any]:
        """Return saved PANDA references for this essay without running a new search.

        Args:
            essay (str): Essay text whose content fingerprint must match the saved record.
            essay_id (str | None): Source ID, which must equal the saved ID.
            timeout (float | None): Accepted for interface compatibility and ignored.

        Returns:
            dict[str, Any]: Saved retrieval status and metadata with references converted
                to a tuple of ReferenceContext objects.

        Raises:
            ValueError: The text or ID does not match the prepared context.
        """
        if text_hash(essay) != self.context["essay_hash"] or essay_id != self.context["essay_id"]:
            raise ValueError("Prepared references belong to another essay")
        return {**self.context["retrieval"], "references": tuple(ReferenceContext(**current_record) for current_record in self.context["references"])}
