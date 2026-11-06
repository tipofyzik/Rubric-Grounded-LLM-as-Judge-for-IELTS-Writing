"""Restore verified runtime assets and prepare the user's local model cache."""
from typing import Any
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())


def digest(path: Any) -> str:
    """Calculate the SHA-256 digest of a file without loading the whole file into memory.

    Args:
        path (Any): Filesystem location to read, validate or write.

    Returns:
        str: The computed result for the supplied inputs.
    """
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):result.update(chunk)
    return result.hexdigest()


def restore(source: Any) -> Any:
    """Validate compatible runtime artifacts before copying missing files into this checkout.

    Args:
        source (Any): Source used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    source = Path(source).resolve()
    aes_path = source / "results/stage3/release.json"
    corpus_path = source / "results/stage4/release.json"
    aes = json.loads(aes_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    aes_root = (aes_path.parent / aes["bundle_path"]).resolve()
    corpus_root = (source / corpus["corpus_path"]).resolve()
    for path in (aes_root,corpus_root):
        if not path.is_relative_to(source):raise ValueError("Artifact directory escapes the source root")
    if digest(aes_root / "manifest.json") != aes["bundle_manifest_sha256"]:raise ValueError("AES manifest mismatch")
    aes_manifest = json.loads((aes_root/"manifest.json").read_text(encoding="utf-8"))
    current_corpus_manifest = json.loads((corpus_root/"manifest.json").read_text(encoding="utf-8"))
    if current_corpus_manifest["corpus_id"] != corpus["corpus_id"]:raise ValueError("Corpus release mismatch")
    expected = {aes_root/name:sha for name,sha in aes_manifest["artifacts"].items()}
    expected[corpus_root/"records.jsonl"] = current_corpus_manifest["records_sha256"]
    for path, sha in expected.items():
        if not path.resolve().is_relative_to(source) or digest(path)!=sha:raise ValueError("Artifact content hash mismatch")
    files = [aes_path,corpus_path,aes_root/"manifest.json",corpus_root/"manifest.json",*expected]
    # Validate all existing destination files before copying, without overwriting another release.
    for path in files:
        destination = ROOT/path.relative_to(source)
        if destination.exists() and digest(destination)!=digest(path):raise FileExistsError("Different artifact already exists: "+str(destination.relative_to(ROOT)))
    for path in files:
        destination = ROOT/path.relative_to(source)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copy2(path,destination)
    return len(files)


def main() -> int:
    """Parse command options and execute this component's command-line operation.

    Returns:
        int: The computed result for the supplied inputs.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-source",type=Path,help="Directory with verified results/stage3 and results/stage4 releases")
    parser.add_argument("--download-assets",action="store_true",help="Fetch user-configured immutable HF artifact repository")
    parser.add_argument("--cache-source",type=Path,help="Optional existing Hugging Face cache; copy model directories only")
    parser.add_argument("--warm-cache",action="store_true")
    parser.add_argument("--offline",action="store_true")
    current_arguments=parser.parse_args()
    os.environ["HF_HOME"]=str(ROOT/".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"]="1" if current_arguments.offline else "0"
    if current_arguments.artifact_source and current_arguments.download_assets:parser.error("Choose one artifact source")
    if current_arguments.download_assets:
        from public_demo.Runtime import restore_artifacts
        names=("HF_ARTIFACT_REPO","HF_ARTIFACT_REVISION","HF_ARTIFACT_MANIFEST_SHA256","HF_TOKEN")
        if any(not os.environ.get(name) for name in names):raise ValueError("Configure "+", ".join(names))
        source=restore_artifacts(ROOT,{"artifact_repo":os.environ[names[0]],"artifact_revision":os.environ[names[1]],"artifact_manifest_sha256":os.environ[names[2]]})
        print(json.dumps({"runtime_files":restore(source)}))
    if current_arguments.artifact_source:print(json.dumps({"runtime_files":restore(current_arguments.artifact_source)}))
    if current_arguments.cache_source:
        for source in (current_arguments.cache_source/"hub").glob("models--*"):
            if source.is_dir():shutil.copytree(source,ROOT/".cache/huggingface/hub"/source.name,dirs_exist_ok=True)
        print("Model cache copied; account tokens are excluded")
    if current_arguments.warm_cache:
        from transformers import AutoTokenizer
        from sentence_transformers import SentenceTransformer
        from ielts_judge.ModelRegistry import ModelRegistry
        profile=ModelRegistry().require_enabled(os.environ.get("IELTS_MODEL_KEY","qwen25_1p5b"))
        AutoTokenizer.from_pretrained(profile.repo_id,revision=profile.revision,token=os.environ.get("HF_TOKEN"))
        release=json.loads((ROOT/"results/stage4/release.json").read_text(encoding="utf-8"))
        embedding=json.loads((ROOT/release["corpus_path"]/"manifest.json").read_text(encoding="utf-8"))["embedding"]
        SentenceTransformer(embedding["model_id"],revision=embedding["revision"],device="cpu",token=os.environ.get("HF_TOKEN"))
        print("Pinned judge tokenizer and BGE model are ready")
    return 0


if __name__=="__main__":raise SystemExit(main())
