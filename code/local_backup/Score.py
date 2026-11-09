"""Run the accepted scoring service once, without saving submission text."""
from typing import Any

from pathlib import Path
import sys

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]

import argparse
import json
from pathlib import Path
import sys

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]

import requests
from local_backup.Runtime import LauncherError, RemoteJudge, configure_environment, load_service, preflight, read_access
from ielts_judge.Prompts import CONTEXT_SETUPS, submission_parts

configure_environment()


def main(command_arguments: Any=None) -> Any:
    """Parse command options and execute this component's command-line operation.

    Args:
        command_arguments (Any): Explicit CLI arguments; None reads the process argument list.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    parser = argparse.ArgumentParser(description="Score one Task/Essay using local AES/BGE and the hosted Qwen judge.")
    parser.add_argument("--task-file", type=Path, help="UTF-8 file containing the Task; requires --essay-file")
    parser.add_argument("--essay-file", type=Path, help="UTF-8 file containing the Essay; requires --task-file")
    parser.add_argument("--setup", choices=tuple(CONTEXT_SETUPS), default="rubric_panda_aes")
    parser.add_argument("--output", type=Path, help="Optional NEW JSON file for scores/metadata; input text is excluded")
    parser.add_argument("--check", action="store_true", help="Check files/packages/Space without loading models or running inference")
    current_arguments = parser.parse_args(command_arguments)
    if (current_arguments.task_file is None) != (current_arguments.essay_file is None):
        parser.error("--task-file and --essay-file must be supplied together")
    try:
        if current_arguments.output is not None and current_arguments.output.exists():
            raise LauncherError("output_exists", "Output file already exists; omit --output or choose a new filename.")
        if current_arguments.task_file is None:
            example = json.loads((ROOT / "code/shared/resources/demo_examples.json").read_text(encoding="utf-8"))["examples"][0]
            task, essay = submission_parts(example)
        else:
            task, essay = submission_parts({"task_prompt": current_arguments.task_file.read_text(encoding="utf-8-sig"),
                                             "essay": current_arguments.essay_file.read_text(encoding="utf-8-sig")})
        values = read_access(current_arguments.setup)
        readiness = preflight(values, current_arguments.setup)
        if current_arguments.check:
            result = readiness
        else:
            print("Loading the local scoring runtime; this may take about 20-30 seconds...", file=sys.stderr)
            judge = RemoteJudge(values, readiness["space"], readiness["judge_endpoint"])
            service = load_service(judge, values, current_arguments.setup)
            result = service.score(model_key=judge.profile.key, task_prompt=task, essay=essay)
        if current_arguments.output is not None:
            current_arguments.output.parent.mkdir(parents=True, exist_ok=True)
            with current_arguments.output.open("x", encoding="utf-8") as output:
                json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["status"] in ("ok", "ready") else 1
    except LauncherError as error:
        result = {"status": "error", "error": {"code": error.code, "message": str(error)}}
    except (OSError, ValueError) as error:
        result = {"status": "error", "error": {"code": "input_or_artifact_error", "type": type(error).__name__,
                  "message": "Check the input files, accepted local artifacts and model cache."}}
    except Exception as error:
        result = {"status": "error", "error": {"code": "startup_failure", "type": type(error).__name__,
                  "message": "Scoring runtime could not start. Use the project .venv and check its dependencies/cache."}}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
