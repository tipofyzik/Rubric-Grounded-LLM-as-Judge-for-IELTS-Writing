"""Build an allowlisted Space probe using the current shared Stage 1 source."""

import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
APP_FILES = ("App.py", "Probe.py", "requirements.txt", "README.md")
CORE_FILES = (
    "code/shared/paths.yaml",
    "code/shared/ielts_judge/__init__.py",
    "code/shared/ielts_judge/Paths.py",
    "code/shared/ielts_judge/Criteria.py",
    "code/shared/ielts_judge/ModelRegistry.py",
    "code/shared/ielts_judge/Prompts.py",
    "code/shared/ielts_judge/Scores.py",
    "code/shared/ielts_judge/llm/__init__.py",
    "code/shared/ielts_judge/llm/Generation.py",
    "code/shared/ielts_judge/llm/JsonEosBlocker.py",
    "code/shared/ielts_judge/llm/Judge.py",
    "code/shared/resources/ielts_criteria.yaml",
    "code/shared/resources/model_registry.yaml",
)


def build_bundle(output_root: Path | None = None) -> tuple[Path, Path]:
    """Copy only approved runtime files, preserving the Stage 0 artifacts.

    Args:
        output_root (Path | None): Output root used by this operation.

    Returns:
        tuple[Path, Path]: The computed result for the supplied inputs.
    """
    output_root = Path(output_root) if output_root is not None else ROOT / "TEMP"
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "ielts_stage1_space"
    sources = {name: ROOT / "code/offline_research/legacy/stage1_space" / name for name in APP_FILES}
    sources.update({name: ROOT / name for name in CORE_FILES})
    hashes = {name: hashlib.sha256(source.read_bytes()).hexdigest()
              for name, source in sorted(sources.items())}
    manifest = {
        "purpose": "Stage 1 real-judge manual smoke; not the hybrid demo.",
        "source_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "files": hashes,
    }
    for name, source in sources.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    manifest_name = "BUILD_MANIFEST.json"
    (destination / manifest_name).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    archive_path = output_root / "ielts_stage1_space.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted((*sources, manifest_name)):
            archive.write(destination / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Stage 1 Space archive failed CRC validation.")
        stored = json.loads(archive.read(manifest_name))
        for name, expected_hash in stored["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected_hash:
                raise ValueError(f"Source hash mismatch: {name}")
    return destination, archive_path


if __name__ == "__main__":
    folder, archive = build_bundle()
    print(f"Upload folder contents: {folder}")
    print(f"ZIP: {archive}")
