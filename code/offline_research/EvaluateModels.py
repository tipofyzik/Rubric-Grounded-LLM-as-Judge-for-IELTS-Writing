"""Run standalone model evaluation; never call or deploy the hosted service."""
from typing import Any

from pathlib import Path
import sys

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]


def main(command_arguments: Any=None) -> int:
    """Prepare inputs, run offline inference, rebuild reports or run all three.

    Inference resumes an existing compatible run. Report mode reads the saved
    run and can regenerate plots without loading a judge. The export option
    creates a results ZIP.

    Args:
        command_arguments (Any): Command-line argument strings, or None to use sys.argv.

    Returns:
        int: Zero after the requested operation succeeds.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "code/offline_research/notebooks/ModelEvaluationConfig.yaml")
    parser.add_argument("--mode", choices=("prepare", "infer", "report", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/evaluation/runs")
    parser.add_argument("--resume-run", type=Path, help="Explicit existing run; settings and source must match")
    parser.add_argument("--max-jobs", type=int, help="Stop after this many new jobs; resume later")
    parser.add_argument("--export", action="store_true", help="Create a new results ZIP after this invocation")
    current_arguments = parser.parse_args(command_arguments)
    if current_arguments.mode in ("infer", "report") and current_arguments.resume_run is None:
        parser.error("infer/report require --resume-run; use prepare/all for a new run")
    cache = ROOT / ".cache/huggingface"
    os.environ.setdefault("HF_HOME", str(cache if cache.exists() else current_arguments.output_root / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from offline_research.evaluation.Config import load_config
    from offline_research.evaluation.Runtime import initialize_run, prepare_run_context, infer_models, report_run, export_run, read_run_context
    config = load_config(current_arguments.config, ROOT)
    if current_arguments.mode == "report":
        run = current_arguments.resume_run.resolve()
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        contexts = read_run_context(run, manifest)
        print(f"Report: {report_run(run, contexts, manifest, config['report'])}", flush=True)
        if current_arguments.export:
            print(f"Export: {export_run(run)}", flush=True)
        return 0
    run, frame, experiment, manifest, paths = initialize_run(ROOT, config, current_arguments.output_root, current_arguments.resume_run)
    print(f"Run: {run}\nSelected essays: {len(frame)}; judge jobs: {len(frame) * len(config['models']) * len(config['setups']) * config['evaluation']['repeats']}", flush=True)
    if current_arguments.mode in ("infer", "report") and not (run / "context_release.json").is_file():
        raise ValueError("Run has no prepared context; execute prepare first")
    contexts = prepare_run_context(run, paths, frame, experiment, manifest, config["runtime"])
    if current_arguments.mode in ("infer", "all"):
        try:
            count = infer_models(ROOT, config, run, frame, contexts, manifest, max_jobs=current_arguments.max_jobs)
            print(f"New saved jobs: {count}", flush=True)
        except BaseException:
            if current_arguments.export:
                print(f"Partial export: {export_run(run)}", flush=True)
            raise
    if current_arguments.mode in ("report", "all"):
        print(f"Report: {report_run(run, contexts, manifest, config['report'])}", flush=True)
    if current_arguments.export:
        print(f"Export: {export_run(run)}", flush=True)
    print(json.dumps({"run_directory": str(run), "service_evaluation": "deferred"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
