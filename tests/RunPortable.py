"""Run offline CPU contract checks from a checkout without private artifacts."""
from typing import Any
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
OUTPUT = ROOT / ".cache/checks"
MODULES = ("TestModelEvaluation", "TestJudge", "TestScoringService", "TestRetrievalRecovery", "TestLLMResponseObtainer")


def main() -> Any:
    """Parse command options and execute this component's command-line operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MPLBACKEND="Agg",
                      LANGSMITH_TRACING="false", TOKENIZERS_PARALLELISM="false")
    sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tests/kaggle"), str(ROOT / "code/shared")]
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    with (OUTPUT / "checks.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    summary = {"modules": MODULES, "tests": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped),
               "scope": "Offline CPU contracts and tiny locally initialized models; no pretrained weights, source datasets, .env, hosted service or GPU benchmark."}
    (OUTPUT / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not result.wasSuccessful():
        print((OUTPUT / "checks.log").read_text(encoding="utf-8"), flush=True)
    return 0 if result.testsRun and result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
