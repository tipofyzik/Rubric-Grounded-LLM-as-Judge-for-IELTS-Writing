"""Load project filesystem locations from the root paths.yaml file."""

from pathlib import Path
from pathlib import PureWindowsPath

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS_CONFIG_PATH = PROJECT_ROOT / "code" / "paths.yaml"


def _resolve_paths(value, root=PROJECT_ROOT):
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


def load_paths(project_root=None):
    """Resolve a selected checkout's configuration, including notebook bundles."""
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    with (root / "code/paths.yaml").open("r", encoding="utf-8") as paths_file:
        paths = yaml.safe_load(paths_file) or {}
    for section in ("datasets", "external", "resources", "results", "evaluation_sources"):
        paths[section] = _resolve_paths(paths.get(section, {}), root)
    return paths

PATHS = load_paths()
