"""Launch the local IELTS demo with the accepted hybrid runtime."""

import argparse
from pathlib import Path
import sys

from ielts_judge.app_runtime import (
    LauncherError,
    RemoteJudge,
    configure_environment,
    load_service,
    preflight,
    read_access,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check artifacts and the HF endpoint without inference",
    )
    args = parser.parse_args(argv)
    configure_environment()
    try:
        values = read_access("rubric_panda_aes")
        ready = preflight(values, "rubric_panda_aes")
        if args.check:
            import json

            print(json.dumps(ready, indent=2))
            return 0
        print(
            "Loading AES, retrieval encoder and judge tokenizer from the local cache...",
            flush=True,
        )
        judge = RemoteJudge(values, ready["space"])
        service = load_service(judge, values, "rubric_panda_aes")
    except LauncherError as error:
        print(f"Startup failed ({error.code}): {error}", file=sys.stderr)
        return 1
    except ImportError as error:
        prepared_python = (
            Path(__file__).resolve().parents[1]
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

    from ielts_judge.ui import build_demo

    demo = build_demo(service)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        show_error=False,
        state_session_capacity=32,
        enable_monitoring=False,
        blocked_paths=[str(Path(__file__).resolve().parents[1])],
        footer_links=[],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
