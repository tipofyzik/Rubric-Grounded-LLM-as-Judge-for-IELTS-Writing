"""Explicit startup wiring for accepted AES/corpus releases; no loading on import."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from .aes.inference import load_bundle
from .database import ReferenceRetriever, SupabaseRPC
from .panda.corpus import Corpus, Encoder
from .prompts import CONTEXT_SETUPS
from .scoring import ScoringService


def load_scoring_service(judge, *, aes_release, corpus_release, project_root,
                         supabase_url, supabase_key, setup="rubric_panda_aes", limits=None,
                         cpu_worker=None):
    """Load once on CPU, preserving the trained AES and pinned embedding profile."""
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
