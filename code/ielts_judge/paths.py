"""Load project filesystem locations from the root paths.yaml file."""

from pathlib import Path
from pathlib import PureWindowsPath

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS_CONFIG_PATH = PROJECT_ROOT / "code" / "paths.yaml"


def _resolve_paths(value):
    if isinstance(value, dict):
        return {key: _resolve_paths(item) for key, item in value.items()}
    if isinstance(value, str):
        path = Path(value)
        is_absolute = (
            path.is_absolute()
            or value.startswith("/")
            or PureWindowsPath(value).is_absolute()
        )
        return str(path if is_absolute else (PROJECT_ROOT / path).resolve())
    return value


with PATHS_CONFIG_PATH.open("r", encoding="utf-8") as paths_file:
    PATHS = yaml.safe_load(paths_file) or {}

for _section in (
    "datasets", "external", "resources", "results", "evaluation_sources"
):
    PATHS[_section] = _resolve_paths(PATHS.get(_section, {}))
