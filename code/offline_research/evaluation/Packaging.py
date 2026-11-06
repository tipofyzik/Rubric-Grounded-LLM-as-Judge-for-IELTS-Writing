"""Standard-library-only Kaggle bootstrap and verified result restoration."""
from typing import Any

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from uuid import uuid4
import zipfile


def sha256_file(path: Any) -> Any:
    """Sha256 file.

    Args:
        path (Any): Filesystem location to read, validate or write.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_id(files: Any) -> Any:
    """Inventory id.

    Args:
        files (Any): Files used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    value = {"schema_version": "model-evaluation-bundle-v1", "files": files}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def safe_relative(name: Any) -> Any:
    """Safe relative.

    Args:
        name (Any): Name used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or str(path) != name:
        raise ValueError(f"Unsafe archive/inventory path: {name}")
    return path


def validate_inventory(files: Any) -> None:
    """Validate inventory.

    Args:
        files (Any): Files used by this operation.

    Returns:
        None: This operation returns no value.
    """
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing file inventory")
    for name, digest in files.items():
        safe_relative(name)
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Invalid file digest")


def copy_verified(source: Any, destination: Any, digest: Any) -> None:
    """Copy verified.

    Args:
        source (Any): Source used by this operation.
        destination (Any): Destination used by this operation.
        digest (Any): Digest used by this operation.

    Returns:
        None: This operation returns no value.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    if sha256_file(destination) != digest:
        raise ValueError(f"File hash mismatch: {destination.name}")


def extract_verified(archive: Any, destination: Any, files: Any, prefix: str="") -> None:
    """Extract verified.

    Args:
        archive (Any): Archive used by this operation.
        destination (Any): Destination used by this operation.
        files (Any): Files used by this operation.
        prefix (str): Prefix used by this operation.

    Returns:
        None: This operation returns no value.
    """
    expected = {prefix + name for name in files}
    actual = [name for name in archive.namelist() if not name.endswith("/")]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Archive members differ from inventory")
    for name, digest in files.items():
        safe_relative(name)
        info = archive.getinfo(prefix + name)
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("Archive symlinks are not allowed")
        with archive.open(info) as source:
            copy_verified(source, destination / name, digest)


def prepare_project(input_root: Any, work_root: Any, manifest_path: Any=None, expected_bundle_id: Any=None) -> Any:
    """Prepare project.

    Args:
        input_root (Any): Input root used by this operation.
        work_root (Any): Work root used by this operation.
        manifest_path (Any): Manifest path used by this operation.
        expected_bundle_id (Any): Expected bundle id used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    candidates = [Path(manifest_path)] if manifest_path is not None else list(Path(input_root).rglob("ModelEvaluationBundleManifest.json"))
    if len(candidates) != 1:
        raise ValueError(f"Attach one new evaluation Dataset or select manifest_path explicitly; found {len(candidates)}")
    path = candidates[0]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "model-evaluation-bundle-v1":
        raise ValueError("This is not a model evaluation Dataset")
    validate_inventory(manifest["files"])
    bundle_id = inventory_id(manifest["files"])
    if manifest.get("bundle_id") != bundle_id or (expected_bundle_id is not None and expected_bundle_id != bundle_id):
        raise ValueError("Notebook and evaluation Dataset belong to different bundles")
    project = Path(work_root).resolve() / ("evaluation-project-" + bundle_id[:12] + "-" + uuid4().hex[:10])
    project.mkdir(parents=True, exist_ok=False)
    archive_name = manifest.get("archive_name", "model_evaluation_bundle.zip")
    safe_relative(archive_name)
    archive_path = path.parent / archive_name
    if archive_path.is_file():
        if sha256_file(archive_path) != manifest["archive_sha256"]:
            raise ValueError("Evaluation archive hash mismatch")
        with zipfile.ZipFile(archive_path) as archive:
            extract_verified(archive, project, manifest["files"])
    else:
        anchor = "code/shared/paths.yaml" if "code/shared/paths.yaml" in manifest["files"] else next(iter(manifest["files"]))
        depth = len(PurePosixPath(anchor).parts)
        roots = {match.parents[depth - 1].resolve() for match in path.parent.rglob(anchor)}
        roots = [candidate for candidate in roots if all((candidate / name).is_file() for name in manifest["files"])]
        if len(roots) != 1:
            raise ValueError(f"Expected one complete unpacked evaluation input tree; found {len(roots)}")
        input_files_root = roots[0]
        for name, digest in manifest["files"].items():
            source_path = (input_files_root / name).resolve()
            if not source_path.is_relative_to(input_files_root):
                raise ValueError("Input file escapes Dataset directory")
            with source_path.open("rb") as source:
                copy_verified(source, project / name, digest)
    (project / "ModelEvaluationBundleManifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return project


def restore_export(archive_path: Any, work_root: Any) -> Any:
    """Restore export.

    Args:
        archive_path (Any): Archive path used by this operation.
        work_root (Any): Work root used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    with zipfile.ZipFile(archive_path) as archive:
        manifests = [name for name in archive.namelist() if name.endswith("/export_manifest.json")]
        if len(manifests) != 1:
            raise ValueError("Expected exactly one evaluation export manifest")
        manifest_name = manifests[0]
        safe_relative(manifest_name)
        manifest = json.loads(archive.read(manifest_name))
        if manifest.get("schema_version") != "model-evaluation-export-v1":
            raise ValueError("Unsupported evaluation export")
        files = manifest["files"]
        validate_inventory(files)
        prefix = manifest_name.removesuffix("export_manifest.json")
        expected = {prefix + name for name in files} | {manifest_name}
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != expected:
            raise ValueError("Unexpected files in evaluation export")
        restored = Path(work_root).resolve() / ("restored-run-" + uuid4().hex)
        restored.mkdir(parents=True, exist_ok=False)
        for name, digest in files.items():
            info = archive.getinfo(prefix + name)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not allowed")
            with archive.open(info) as source:
                copy_verified(source, restored / name, digest)
        (restored / "restored_export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return restored
