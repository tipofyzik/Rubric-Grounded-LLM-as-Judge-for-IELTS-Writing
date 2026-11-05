"""Prepare a small Kaggle fixture, rebuild its ZIP and run local notebook checks."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.dont_write_bytecode = True


def main():
    module_root = Path(__file__).resolve().parent
    project_root = module_root.parent
    cache = module_root / ".cache"
    scratch = cache / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    for variable in ("TEMP", "TMP", "TMPDIR"):
        os.environ[variable] = str(scratch)
    tempfile.tempdir = str(scratch)
    for variable, relative in {
        "XDG_CACHE_HOME": ".", "PIP_CACHE_DIR": "pip",
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
        "HF_ASSETS_CACHE": "huggingface/assets", "HF_DATASETS_CACHE": "huggingface/datasets",
        "TORCH_HOME": "torch", "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor", "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda", "NUMBA_CACHE_DIR": "numba",
        "MPLCONFIGDIR": "matplotlib", "WANDB_DIR": "wandb", "WANDB_CACHE_DIR": "wandb-cache",
    }.items():
        os.environ[variable] = str(cache / relative)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, str(project_root / "code"))

    import yaml
    from build_bundle import build_bundle
    from ielts_judge.data.DatasetTransformer import DatasetTransformer
    from ielts_judge.paths import load_paths

    output = module_root / "artifacts/tests"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "Results.json"
    summary_path.unlink(missing_ok=True)
    log_path = output / "Checks.log"
    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        configuration = yaml.safe_load((project_root / "code/config.yaml").read_text(encoding="utf-8"))
        transformer = DatasetTransformer(
            load_paths(project_root)["datasets"], selection=configuration.get("dataset", {}).get("selection"),
        )
        prepared = transformer.prepare_and_save(output / "prepared", limit_per_split=12)
        build_bundle(prepared, allow_partial=True)
        suite = unittest.defaultTestLoader.discover(str(module_root / "tests"), pattern="Test*.py")
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    summary = {
        "tests": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped), "log": str(log_path),
        "limitations": "Local CPU checks; no pretrained model downloaded, GPU training and Kaggle hosting not tested.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not result.wasSuccessful():
        print(log_path.read_text(encoding="utf-8")[-12000:])
    return not result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(main())
