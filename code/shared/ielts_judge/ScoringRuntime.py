"""Explicit startup wiring for accepted AES/corpus releases; no loading on import."""
from typing import Any

import hashlib
import importlib.metadata
import json
from pathlib import Path

from ielts_judge.aes.Inference import load_bundle
from ielts_judge.Database import ReferenceRetriever, SupabaseRPC
from ielts_judge.panda.Corpus import Corpus, Encoder
from ielts_judge.Prompts import CONTEXT_SETUPS
from ielts_judge.Scoring import ScoringService


def load_scoring_service(judge: Any, *, aes_release: str | Path, corpus_release: str | Path, project_root: str | Path,
                         supabase_url: str, supabase_key: str, setup: str="rubric_panda_aes", limits: Any=None,
                         cpu_worker: Any=None) -> ScoringService:
    """Assemble the scoring service with the CPU components required by its setup.

    Args:
        judge (Any): Loaded judge adapter providing token counting and generation.
        aes_release (str | Path): Path to the AES descriptor, used when AES is enabled.
        corpus_release (str | Path): Path to the PANDA descriptor, used for retrieval.
        project_root (str | Path): Checkout root for paths in the PANDA descriptor.
        supabase_url (str): Supabase project origin, required for retrieval.
        supabase_key (str): API key authorized to search PANDA, required for retrieval.
        setup (str): Context setup selecting rubric, PANDA references and AES guidance.
        limits (Any): ScoringLimits, or None to use the service defaults.
        cpu_worker (Any): Optional worker that loads and owns CPU models in a child process.

    Returns:
        ScoringService: Ready service with the selected adapters and release metadata.

    Raises:
        ValueError: The setup, release or runtime settings are incompatible.
    """
    if setup not in CONTEXT_SETUPS:
        raise ValueError("Unknown context setup")
    _, use_references, use_aes = CONTEXT_SETUPS[setup]
    versions = {"langchain_core": importlib.metadata.version("langchain-core"),
                "tokenizer_id": judge.profile.repo_id, "tokenizer_revision": judge.profile.revision}
    aes = retriever = None
    if use_aes:
        release_path = Path(aes_release)
        release = json.loads(release_path.read_text(encoding="utf-8"))
        if release.get("paths_relative_to") != "this_file_parent":
            raise ValueError("Unsupported AES release path convention")
        bundle_path = release_path.parent / release["bundle_path"]
        digest = hashlib.sha256((bundle_path / "manifest.json").read_bytes()).hexdigest()
        if digest != release["bundle_manifest_sha256"]:
            raise ValueError("AES release manifest hash mismatch")
        aes = cpu_worker.load_aes(bundle_path) if cpu_worker is not None else load_bundle(bundle_path, device="cpu")
        versions["aes_bundle_manifest_sha256"] = digest
    if use_references:
        release = json.loads(Path(corpus_release).read_text(encoding="utf-8"))
        corpus = Corpus(Path(project_root) / release["corpus_path"])
        manifest = corpus.manifest
        if release["corpus_id"] != manifest["corpus_id"] or release["count"] != manifest["count"]:
            raise ValueError("Corpus release differs from the verified artifact")
        encoder = cpu_worker.load_encoder(manifest["embedding"]) if cpu_worker is not None else Encoder(manifest["embedding"], device="cpu")
        retriever = ReferenceRetriever(SupabaseRPC(supabase_url, supabase_key), manifest, encoder)
        versions.update(corpus_id=manifest["corpus_id"],
                        embedding_profile_sha256=manifest["embedding_profile_sha256"],
                        encoder=dict(manifest["embedding"]))
    return ScoringService(judge, aes=aes, retriever=retriever, setup=setup, limits=limits, versions=versions)
