"""Read only user runtime settings; source distributions never carry credentials."""
from typing import Any
import os
from pathlib import Path


class LauncherError(RuntimeError):
    """Report launcher failures to the caller."""
    def __init__(self, code: str, message: str) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            code (str): Code used by this operation.
            message (str): Message used by this operation.

        Returns:
            None: This operation returns no value.
        """
        super().__init__(message)
        self.code = code


def read_access(setup: str, *, root: Any) -> Any:
    """Read the runtime settings whitelist with process values taking precedence.

    Args:
        setup (str): Context setup selecting rubric, PANDA references and AES guidance.
        root (Any): Project root used to resolve relative configuration paths.

    Returns:
        Any: The computed result for the supplied inputs.
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
