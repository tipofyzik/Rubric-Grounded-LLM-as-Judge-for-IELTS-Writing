"""Read only user runtime settings; source distributions never carry credentials."""
from typing import Any
import os
from pathlib import Path


class LauncherError(RuntimeError):
    """Report launcher failures to the caller."""
    def __init__(self, code: str, message: str) -> None:
        """Keep the machine-readable launcher code with its message.

        Args:
            code (str): Public error identifier used by CLI and UI callers.
            message (str): Explanation suitable for showing to the operator.
        """
        super().__init__(message)
        self.code = code


def read_access(setup: str, *, root: Any) -> Any:
    """Load the runtime settings whitelist from .env and the process environment.

    Nonempty process values take precedence over file values. HF_TOKEN is
    required. Retrieval setups also require Supabase URL and publishable key.

    Args:
        setup (str): Context setup determining which settings are required.
        root (Any): Project directory that may contain .env.

    Returns:
        Any: Mapping of supported runtime setting names to string values.

    Raises:
        LauncherError: A required setting is missing.
    """
    from .Prompts import CONTEXT_SETUPS
    path = Path(root) / ".env"
    allowed = {"HF_TOKEN", "HF_SPACE_ID", "SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "IELTS_MODEL_KEY"}
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines() if path.is_file() else []:
        if line.strip() and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            if key.strip() in allowed:values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key:os.environ[key] for key in allowed if os.environ.get(key)})
    required = ["HF_TOKEN"] + (["SUPABASE_URL","SUPABASE_PUBLISHABLE_KEY"] if CONTEXT_SETUPS[setup][1] else [])
    missing = [key for key in required if not values.get(key)]
    if missing:raise LauncherError("missing_access", "Configure .env or environment variables: "+", ".join(missing))
    return values
