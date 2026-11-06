"""Run the accepted scoring service once, without saving submission text."""

import argparse
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "code/paths.yaml").is_file())
sys.path.insert(0, str(ROOT / "code"))

import requests
from ielts_judge.app_runtime import (
    LauncherError, RemoteJudge, configure_environment, load_service, preflight, read_access,
)
from ielts_judge.prompts import CONTEXT_SETUPS, submission_parts

configure_environment()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score one Task/Essay using local AES/BGE and the hosted Qwen judge.")
    parser.add_argument("--task-file", type=Path, help="UTF-8 file containing the Task; requires --essay-file")
    parser.add_argument("--essay-file", type=Path, help="UTF-8 file containing the Essay; requires --task-file")
    parser.add_argument("--setup", choices=tuple(CONTEXT_SETUPS), default="rubric_panda_aes")
    parser.add_argument("--output", type=Path, help="Optional NEW JSON file for scores/metadata; input text is excluded")
    parser.add_argument("--check", action="store_true", help="Check files/packages/Space without loading models or running inference")
    args = parser.parse_args(argv)
    if (args.task_file is None) != (args.essay_file is None):
        parser.error("--task-file and --essay-file must be supplied together")
    try:
        if args.output is not None and args.output.exists():
            raise LauncherError("output_exists", "Output file already exists; omit --output or choose a new filename.")
        if args.task_file is None:
            example = json.loads((ROOT / "results/stage5/2026-10-31/verification/smoke_submission.json").read_text(encoding="utf-8"))
            task, essay = submission_parts(example)
        else:
            task, essay = submission_parts({"task_prompt": args.task_file.read_text(encoding="utf-8-sig"),
                                             "essay": args.essay_file.read_text(encoding="utf-8-sig")})
        values = read_access(args.setup)
        readiness = preflight(values, args.setup)
        if args.check:
            result = readiness
        else:
            print("Loading the local scoring runtime; this may take about 20-30 seconds...", file=sys.stderr)
            judge = RemoteJudge(values, readiness["space"])
            service = load_service(judge, values, args.setup)
            result = service.score(model_key="qwen25_1p5b", task_prompt=task, essay=essay)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
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
