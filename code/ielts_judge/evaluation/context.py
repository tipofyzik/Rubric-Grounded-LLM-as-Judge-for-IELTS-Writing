"""Prepare real AES/BGE context once, using the accepted train corpus offline."""

from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import numpy as np

from ..criteria import PUBLIC_CRITERION_IDS
from ..data.preparation import file_hash, text_hash
from ..prompts import ReferenceContext


def asset_paths(root, config):
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


def nearest_references(matrix, records, query_vector, *, essay_id, essay_hash, k):
    from ..panda.corpus import vector
    query_vector = vector(query_vector)
    similarities = matrix @ query_vector / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector))
    eligible = np.array([r["essay_id"] != essay_id and r["essay_hash"] != essay_hash for r in records])
    order = sorted(np.flatnonzero(eligible), key=lambda i: (-float(similarities[i]), records[i]["essay_id"]))[:k]
    references = []
    for index in order:
        row = records[index]
        references.append(asdict(ReferenceContext(
            essay_id=row["essay_id"], task_prompt=row["task_prompt"], essay=row["essay"],
            scores={c: row["criteria"][c]["score"] for c in PUBLIC_CRITERION_IDS}, insights=row["insights"],
        )))
    return references, [float(similarities[i]) for i in order]


def prepare_contexts(paths, frame, experiment, *, device="cpu", batch_size=8, k=1):
    from ..aes.OrdinalRegression import OrdinalRegression, token_lengths
    from ..panda.corpus import Corpus, Encoder
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
    if set(frame.essay_id).intersection(r["essay_id"] for r in source) or set(frame.essay_hash).intersection(r["essay_hash"] for r in source):
        raise ValueError("Evaluation essays must not be reference train records")
    matrix = np.array([r["embedding"] for r in source], dtype=np.float32)
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
            essay_id=row.essay_id, essay_hash=row.essay_hash, k=k)
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
    def __init__(self, context):
        self.context = context

    def predict_essay(self, essay):
        if text_hash(essay) != self.context["essay_hash"]:
            raise ValueError("Prepared AES context belongs to another essay")
        return self.context["aes"]


class PreparedRetriever:
    def __init__(self, context):
        self.context = context

    def retrieve(self, essay, *, essay_id=None, timeout=None):
        if text_hash(essay) != self.context["essay_hash"] or essay_id != self.context["essay_id"]:
            raise ValueError("Prepared references belong to another essay")
        return {**self.context["retrieval"], "references": tuple(ReferenceContext(**r) for r in self.context["references"])}
