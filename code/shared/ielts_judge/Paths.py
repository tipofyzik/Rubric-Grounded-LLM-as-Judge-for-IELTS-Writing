"""Load project filesystem locations from the root paths.yaml file."""
from typing import Any

import json
from pathlib import Path
from pathlib import PureWindowsPath

import yaml


PROJECT_ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
PATHS_CONFIG_PATH = PROJECT_ROOT / "code/shared/paths.yaml"
RELEASE_PATHS = {"aes": Path("results/aes/release.json"), "panda": Path("results/panda/release.json")}
LEGACY_RELEASE_PATHS = {"aes": Path("results/stage3/release.json"), "panda": Path("results/stage4/release.json")}


def find_release(root: Path, name: str) -> Path:
    """Select a release from the artifact inventory or the local checkout."""
    current = root / RELEASE_PATHS[name]
    legacy = root / LEGACY_RELEASE_PATHS[name]
    inventory = root / "ARTIFACT_MANIFEST.json"
    if inventory.is_file():
        files = json.loads(inventory.read_text(encoding="utf-8"))["files"]
        for path in (current, legacy):
            if path.relative_to(root).as_posix() in files:
                return path
        raise FileNotFoundError(f"Artifact inventory is missing the {name} release")
    return current if current.exists() or not legacy.is_file() else legacy


def _resolve_paths(value: Any, root: Any=PROJECT_ROOT) -> Any:
    """Resolve paths.

    Args:
        value (Any): Value used by this operation.
        root (Any): Project root used to resolve relative configuration paths.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if isinstance(value, dict):
        return {key: _resolve_paths(item, root) for key, item in value.items()}
    if isinstance(value, str):
        path = Path(value)
        is_absolute = (
            path.is_absolute()
            or value.startswith("/")
            or PureWindowsPath(value).is_absolute()
        )
        return str(path if is_absolute else (root / path).resolve())
    return value


def load_paths(project_root: Any=None) -> Any:
    """Resolve a selected checkout's configuration, including notebook bundles.

    Args:
        project_root (Any): Selected source checkout used for configuration and artifacts.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    with (root / "code/shared/paths.yaml").open("r", encoding="utf-8") as paths_file:
        paths = yaml.safe_load(paths_file) or {}
    for section in ("datasets", "external", "resources", "results", "evaluation_sources"):
        paths[section] = _resolve_paths(paths.get(section, {}), root)
    return paths

PATHS = load_paths()
