"""Prepare allowlisted public source and private runtime artifacts."""
from typing import Any

import hashlib
import json
from pathlib import Path
import shutil
import sys

import yaml

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from Runtime import file_hash

CORE = (
    "__init__.py", "Paths.py", "Criteria.py", "ModelRegistry.py", "Localization.py",
    "Prompts.py", "Scores.py", "Scoring.py", "ScoringRuntime.py", "Database.py", "Ui.py",
    "llm/__init__.py", "llm/Judge.py", "llm/Generation.py", "llm/JsonEosBlocker.py",
    "aes/__init__.py", "aes/Inference.py", "aes/OrdinalRegression.py",
    "data/__init__.py", "data/Preparation.py", "data/Experiment.py",
    "panda/__init__.py", "panda/Artifacts.py", "panda/Corpus.py",
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Serialize the supplied value to a UTF-8 JSON document.
    
    Args:
        path (Path): Filesystem location to read, validate or write.
        value (dict[str, Any]): Value used by this operation.
    
    Returns:
        None: This operation returns no value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_artifacts(destination: Path) -> dict[str, Any]:
    """Copy the accepted AES/corpus inputs and record their immutable content hashes.
    
    Args:
        destination (Path): Destination used by this operation.
    
    Returns:
        dict[str, Any]: The computed result for the supplied inputs.
    """
    sources = [ROOT / "results/stage3/release.json", ROOT / "results/stage4/release.json"]
    aes = json.loads(sources[0].read_text(encoding="utf-8"))
    corpus = json.loads(sources[1].read_text(encoding="utf-8"))
    for directory in (sources[0].parent / aes["bundle_path"], ROOT / corpus["corpus_path"]):
        sources.extend(candidate_path for candidate_path in directory.rglob("*") if candidate_path.is_file())
    files = {}
    for source in sources:
        name = source.relative_to(ROOT).as_posix()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or file_hash(target) != file_hash(source):
            shutil.copyfile(source, target)
        files[name] = file_hash(target)
    manifest = {"schema_version": "ielts-hosted-artifacts-v1", "files": files,
                "corpus_id": corpus["corpus_id"], "aes_manifest_sha256": aes["bundle_manifest_sha256"]}
    write_json(destination / "ARTIFACT_MANIFEST.json", manifest)
    return manifest


def prepare_source(destination: Path, *, artifact_repo: str, artifact_revision: str, artifact_manifest_sha256: str) -> dict[str, Any]:
    """Build the explicit public Space file set and pin its private runtime-artifact release.
    
    Args:
        destination (Path): Destination used by this operation.
        artifact_repo (str): Artifact repo used by this operation.
        artifact_revision (str): Artifact revision used by this operation.
        artifact_manifest_sha256 (str): Artifact manifest sha256 used by this operation.
    
    Returns:
        dict[str, Any]: The computed result for the supplied inputs.
    """
    sources = {candidate_path: HERE / candidate_path for candidate_path in ("App.py", "Runtime.py", "CpuWorker.py", "requirements.txt", "README.md")}
    sources.update({"code/shared/ielts_judge/" + candidate_path: ROOT / "code/shared/ielts_judge" / candidate_path for candidate_path in CORE})
    for name in ("code/shared/paths.yaml", "code/shared/resources/ielts_criteria.yaml",
                 "code/shared/resources/model_registry.yaml", "code/shared/resources/demo_examples.json",
                 "code/shared/resources/locales/en.json", "code/shared/resources/locales/ru.json", "code/shared/resources/locales/de.json"):
        sources[name] = ROOT / name
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    config = yaml.safe_load((ROOT / "code/shared/config.yaml").read_text(encoding="utf-8"))
    (destination / "code/shared/config.yaml").write_text(
        yaml.safe_dump({"retrieval": {"k": config["retrieval"]["k"]}}), encoding="utf-8")
    hashes = {name: file_hash(destination / name) for name in sorted(sources)}
    hashes["code/shared/config.yaml"] = file_hash(destination / "code/shared/config.yaml")
    deployment = {"schema_version": "ielts-hosted-deployment-v1", "artifact_repo": artifact_repo,
                  "artifact_revision": artifact_revision, "artifact_manifest_sha256": artifact_manifest_sha256,
                  "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                  "files": hashes}
    write_json(destination / "DEPLOYMENT.json", deployment)
    (destination / ".gitignore").write_text(".runtime/\n.env\n__pycache__/\n", encoding="utf-8")
    return deployment


