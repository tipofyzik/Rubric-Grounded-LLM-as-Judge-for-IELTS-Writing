"""Build the Kaggle runtime ZIP from shared code and a verified dataset release.

Uses an explicit allowlist of local research fixtures. No network, weights,
secrets or changes to datasets. This is not the public demo deployment bundle.
"""
from typing import Any

import hashlib
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True


def build_bundle(prepared_release: Any=None, *, allow_partial: bool=False, output_directory: Any=None) -> Path:
    """Build bundle.

    Args:
        prepared_release (Any): Prepared release used by this operation.
        allow_partial (bool): Allow partial used by this operation.
        output_directory (Any): Output directory used by this operation.

    Returns:
        Path: The computed result for the supplied inputs.
    """
    root = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
    sys.path[:0] = [str(root / "code"), str(root / "code/shared")]
    from ielts_judge.Paths import load_paths
    from ielts_judge.data.Preparation import load_prepared

    configured_release = Path(load_paths(root)["datasets"]["prepared"])
    prepared_target = configured_release.relative_to(root).as_posix()
    files = {
        "RunConfig.yaml": root / "code/offline_research/notebooks/RunConfig.yaml",
        "requirements.txt": root / "requirements.txt",
    }
    for relative in (
        "code/offline_research/PrepareData.py",
        "code/offline_research/TrainAes.py",
        "code/offline_research/config.yaml",
        "code/shared/paths.yaml", "code/shared/config.yaml",
        "code/shared/resources/ielts_criteria.yaml",
        "code/shared/resources/model_registry.yaml",
        "data/datasets/chillies - IELTS-writing-task-2-evaluation/transformed_dataset.csv",
        "data/datasets/preprocessed_dataset.csv",
        "data/legacy/AES predictions/predictions_table.csv",
    ):
        files[relative] = root / relative
    release = (
        Path(prepared_release).resolve() if prepared_release else configured_release
    )
    if not (release / "manifest.json").is_file():
        raise FileNotFoundError(
            f"Prepared dataset manifest is missing: {release / 'manifest.json'}. "
            "Prepare the configured release with python -B code/offline_research/PrepareData.py, "
            "or select an existing release with --prepared."
        )
    _, _, dataset_manifest = load_prepared(release, allow_partial=allow_partial)
    for name in ("manifest.json", *dataset_manifest["artifacts"]):
        files[prepared_target + "/" + name] = release / name
    for folder, pattern in (
        ("code", "*.py"),
        ("data/legacy/LLM evaluation/results", "*.csv"),
    ):
        for source in (root / folder).rglob(pattern):
            parts = source.relative_to(root).parts
            if folder == "code" and (set(parts).intersection({"__pycache__", "artifacts", ".cache", "tools", "notebooks", "legacy", "public_demo", "local_backup"})):
                continue
            files[source.relative_to(root).as_posix()] = source
    for source in files.values():
        if not source.is_file():
            raise FileNotFoundError(f"Required local test fixture is missing: {source}")
    output = (Path(output_directory).resolve() if output_directory else
              root / "code/offline_research/notebooks/artifacts" / ("tests/bundle" if allow_partial else "bundle"))
    destination = output / "ielts_kaggle_bundle.zip"
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in sorted(files.items()):
            archive.write(source, name)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Kaggle test bundle failed CRC validation.")
    manifest = {
        "purpose": "Kaggle notebook runtime; not a public demo release.",
        "partial_input": dataset_manifest["partial_input"],
        "archive_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "files": {
            name: hashlib.sha256(source.read_bytes()).hexdigest()
            for name, source in sorted(files.items())
        },
    }
    (output / "KaggleBundleManifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared",
        type=Path,
        help="Verified Parquet release; default reads code/shared/paths.yaml",
    )
    parser.add_argument(
        "--output-directory", type=Path, help="Write a separate bundle without replacing an existing release",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a test preview and write to artifacts/tests/bundle instead of artifacts/bundle",
    )
    current_arguments = parser.parse_args()
    print(build_bundle(current_arguments.prepared, allow_partial=current_arguments.allow_partial, output_directory=current_arguments.output_directory))
