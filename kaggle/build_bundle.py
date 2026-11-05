"""Build the Kaggle runtime ZIP from shared code and a verified dataset release.

Uses an explicit allowlist of local research fixtures. No network, weights,
secrets or changes to datasets. This is not the public demo deployment bundle.
"""

import hashlib
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True


def build_bundle(prepared_release=None, *, allow_partial=False, output_directory=None) -> Path:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "code"))
    from ielts_judge.paths import load_paths
    from ielts_judge.data.preparation import load_prepared

    configured_release = Path(load_paths(root)["datasets"]["prepared"])
    prepared_target = configured_release.relative_to(root).as_posix()
    files = {
        "RunConfig.yaml": root / "kaggle/RunConfig.yaml",
        "requirements.txt": root / "requirements.txt",
    }
    for relative in (
        "code/main.py",
        "code/aes.py",
        "code/config.yaml",
        "code/paths.yaml",
        "code/resources/ielts_criteria.yaml",
        "code/resources/model_registry.yaml",
        "code/resources/datasets/chillies - IELTS-writing-task-2-evaluation/transformed_dataset.csv",
        "code/resources/datasets/preprocessed_dataset.csv",
        "code/resources/AES predictions/predictions_table.csv",
    ):
        files[relative] = root / relative
    release = (
        Path(prepared_release).resolve() if prepared_release else configured_release
    )
    if not (release / "manifest.json").is_file():
        raise FileNotFoundError(
            f"Prepared dataset manifest is missing: {release / 'manifest.json'}. "
            "Prepare the configured release with python -B code/main.py, "
            "or select an existing release with --prepared."
        )
    _, _, dataset_manifest = load_prepared(release, allow_partial=allow_partial)
    for name in ("manifest.json", *dataset_manifest["artifacts"]):
        files[prepared_target + "/" + name] = release / name
    for folder, pattern in (
        ("code/ielts_judge", "*.py"),
        ("code/resources/LLM evaluation/results", "*.csv"),
    ):
        for source in (root / folder).rglob(pattern):
            files[source.relative_to(root).as_posix()] = source
    for source in files.values():
        if not source.is_file():
            raise FileNotFoundError(f"Required local test fixture is missing: {source}")
    output = (Path(output_directory).resolve() if output_directory else
              root / "kaggle/artifacts" / ("tests/bundle" if allow_partial else "bundle"))
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
        help="Verified Parquet release; default reads code/paths.yaml",
    )
    parser.add_argument(
        "--output-directory", type=Path, help="Write a separate bundle without replacing an existing release",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a test preview and write to artifacts/tests/bundle instead of artifacts/bundle",
    )
    args = parser.parse_args()
    print(build_bundle(args.prepared, allow_partial=args.allow_partial, output_directory=args.output_directory))
