"""Launch the local IELTS demo with the accepted hybrid runtime."""
from typing import Any

from pathlib import Path
import sys

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]

import argparse
from pathlib import Path
import sys

from local_backup.Runtime import LauncherError, RemoteJudge, configure_environment, load_service, preflight, read_access


def main(command_arguments: Any=None) -> int:
    """Parse command options and execute this component's command-line operation.

    Args:
        command_arguments (Any): Explicit CLI arguments; None reads the process argument list.

    Returns:
        int: The computed result for the supplied inputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check artifacts and the HF endpoint without inference",
    )
    current_arguments = parser.parse_args(command_arguments)
    configure_environment()
    try:
        values = read_access("rubric_panda_aes")
        ready = preflight(values, "rubric_panda_aes")
        if current_arguments.check:
            import json

            print(json.dumps(ready, indent=2))
            return 0
        print(
            "Loading AES, retrieval encoder and judge tokenizer from the local cache...",
            flush=True,
        )
        judge = RemoteJudge(values, ready["space"], ready["judge_endpoint"])
        service = load_service(judge, values, "rubric_panda_aes")
    except LauncherError as error:
        print(f"Startup failed ({error.code}): {error}", file=sys.stderr)
        return 1
    except ImportError as error:
        prepared_python = (
            next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
            / ".venv/Scripts/python.exe"
        )
        print(f"Startup failed (ImportError): {error}", file=sys.stderr)
        print(f"Python interpreter: {sys.executable}", file=sys.stderr)
        print(f"Select this interpreter in your IDE: {prepared_python}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"Startup failed ({type(error).__name__}). Check the accepted artifacts, dependencies and local model cache.",
            file=sys.stderr,
        )
        return 1

    from ielts_judge.Ui import build_demo

    demo = build_demo(service)
    demo.launch(
        server_name="127.0.0.1",
        server_port=current_arguments.port,
        share=False,
        show_error=False,
        state_session_capacity=32,
        enable_monitoring=False,
        blocked_paths=[str(next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file()))],
        footer_links=[],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
