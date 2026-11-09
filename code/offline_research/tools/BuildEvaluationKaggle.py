"""Create a NEW verified evaluation bundle and matching notebook; no uploads."""
from typing import Any

import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]

from ielts_judge.data.Preparation import file_hash, load_prepared
from ielts_judge.data.Experiment import prepare_experiment
from offline_research.evaluation.Config import load_config
from offline_research.evaluation.Context import asset_paths
from offline_research.evaluation.Packaging import inventory_id
from offline_research.evaluation.Runtime import unique_name


def notebook_document(bundle_id: Any=None) -> dict[str, Any]:
    """Notebook document.

    Args:
        bundle_id (Any): Bundle id used by this operation.

    Returns:
        dict[str, Any]: Named result fields and metadata produced by this operation.
    """
    bootstrap = (ROOT / "code/offline_research/evaluation/Packaging.py").read_text(encoding="utf-8")
    cells = []

    def cell(kind: Any, source: Any) -> None:
        """Cell.

        Args:
            kind (Any): Kind used by this operation.
            source (Any): Source used by this operation.

        Returns:
            None: This operation returns no value.
        """
        item = {"cell_type": kind, "id": f"{len(cells):02d}-{kind}", "metadata": {}, "source": source}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", "# IELTS Writing Model Evaluation\n\nNew, independent model benchmark: AES and six Qwen context setups.\nUse the NEW private `ielts-writing-model-evaluation` Dataset. Existing notebooks and Datasets are not replaced.\nNo hosted service or Supabase calls. Service evaluation is deferred. Enable a GPU and Internet before inference.\n")
    cell("code", bootstrap)
    cell("markdown", "## 1. Select input and resume\nAttach the matching new Dataset. Leave the manifest unset if exactly one evaluation Dataset is attached.\nA new project directory is created each time this cell runs.\nFor a later session, attach an exported result ZIP and set RESUME_ARCHIVE.\nUse one notebook/session as writer for each run.\n")
    cell("code", f"EXPECTED_BUNDLE_ID = {bundle_id!r}\nMANIFEST_PATH = None\nRESUME_ARCHIVE = None\nRESUME_RUN = None  # Existing writable run directory in this session, or restored below.\nproject_root = prepare_project('/kaggle/input', '/kaggle/working/model-evaluation-projects',\n                               MANIFEST_PATH, EXPECTED_BUNDLE_ID)\nif RESUME_ARCHIVE is not None:\n    RESUME_RUN = restore_export(RESUME_ARCHIVE, '/kaggle/working/model-evaluation-restored')\nprint('Project:', project_root)\nprint('Resume:', RESUME_RUN)\n")
    cell("markdown", "## 2. Install the pinned evaluation dependencies\nA new virtual environment inherits the host CUDA PyTorch; pinned evaluation packages are installed into the new environment.\nThe exact torch constraint preserves the installed CUDA build. All model work runs in a fresh Python subprocess.\nPackage conflicts stop installation rather than silently changing the requested versions.\n")
    cell("code", "import importlib.metadata\nimport os\nimport subprocess\nimport sys\nimport venv\nenv_path = project_root / '.evaluation-venv'\nvenv.EnvBuilder(with_pip=False, system_site_packages=True).create(env_path)\nEVALUATION_PYTHON = str(env_path / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))\nconstraint = project_root / 'torch-constraint.txt'\nconstraint.write_text('torch==' + importlib.metadata.version('torch') + '\\n', encoding='utf-8')\nsubprocess.check_call([sys.executable, '-m', 'pip', '--python', EVALUATION_PYTHON, 'install', '-r',\n                       str(project_root / 'code/offline_research/requirements-evaluation.txt'), '-c', str(constraint)])\nos.environ['HF_HOME'] = '/kaggle/working/model-evaluation-cache/huggingface'\nos.environ['HF_HUB_CACHE'] = os.environ['HF_HOME'] + '/hub'\nos.environ['TOKENIZERS_PARALLELISM'] = 'false'\nos.environ['LANGSMITH_TRACING'] = 'false'\nos.environ['PYTHONDONTWRITEBYTECODE'] = '1'\n")
    cell("markdown", "## 3. Experiment and plot configuration\nEdit the YAML below. Each changed experiment receives a new run.\nFor a technical pilot set sample_size to 10 or 20 before starting; the default is the full test pool.\n`report.plots` chooses graph types; `[]` still exports all numeric tables. There is no smoothing or half-band accuracy.\nChanging only report settings permits re-reporting an existing run into a new report directory.\n")
    cell("code", "CONFIG_YAML = (project_root / 'code/offline_research/notebooks/ModelEvaluationConfig.yaml').read_text(encoding='utf-8')\nprint(CONFIG_YAML)\n# Replace CONFIG_YAML with an edited triple-quoted YAML string if needed.\n# Plot types: metric_comparison, metric_bars, metric_heatmap, calibration, calibration_by_prediction, bias,\n# error_histogram, error_histogram_std, error_ridge, confusion_matrix, score_distribution,\n# coverage, prompt_comparison, repeatability, latency.\n")
    cell("markdown", "## 4. Run models or regenerate reports\nModes: all, prepare (real AES/BGE only), infer (needs RESUME_RUN), report (no model loading; needs RESUME_RUN).\nMAX_JOBS optionally limits NEW judge jobs per invocation. Failed jobs remain recorded; resume runs only missing jobs.\nThe command prints the run path: set RESUME_RUN to that path before continuing it.\nThe full default matrix is 1,635 essays x 6 setups = 9,810 judge jobs.\n")
    cell("code", "MODE = 'all'\nMAX_JOBS = None\nconfig_path = project_root / ('evaluation-config-' + uuid4().hex + '.yaml')\nconfig_path.write_text(CONFIG_YAML, encoding='utf-8')\ncommand = [EVALUATION_PYTHON, '-B', str(project_root / 'code/offline_research/EvaluateModels.py'),\n           '--config', str(config_path), '--mode', MODE, '--export',\n           '--output-root', '/kaggle/working/model-evaluation-runs']\nif RESUME_RUN is not None:\n    command += ['--resume-run', str(RESUME_RUN)]\nif MAX_JOBS is not None:\n    command += ['--max-jobs', str(MAX_JOBS)]\nsubprocess.check_call(command)\n")
    cell("markdown", "## 5. Download results\nSave the notebook version with outputs and download the NEW ZIP printed above.\nExports include manifests, append-only predictions/raw completions, AES/reference context, numeric tables and selected figures.\nKaggle latency describes this model experiment. Public service latency/queue/restart checks are deferred.\n")
    cell("markdown", "## 6. Calibration curves and a separate results ZIP\nThis cell exports existing results without running models. The archive is placed in /kaggle/working/IELTS_Evaluation_Results with a readable name.\n")
    cell("code", (ROOT / "code/offline_research/evaluation/Results.py").read_text(encoding="utf-8") + "\n\nRESULTS_RUN = RESUME_RUN\nif RESULTS_RUN is None:\n    candidates = list(Path('/kaggle/working/model-evaluation-runs').rglob('run_manifest.json'))\n    if not candidates:\n        raise ValueError('Set RESULTS_RUN to the printed Run directory')\n    RESULTS_RUN = max(candidates, key=lambda p: p.stat().st_mtime).parent\nshow_results(RESULTS_RUN)\n")
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "model_evaluation_bundle_id": bundle_id}, "cells": cells}


def build_bundle(output_root: Any=None) -> Any:
    """Build bundle.

    Args:
        output_root (Any): Output root used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    config = load_config(ROOT / "code/offline_research/notebooks/ModelEvaluationConfig.yaml", ROOT)
    paths = asset_paths(ROOT, config)
    train, test, _ = load_prepared(paths["dataset"])
    from ielts_judge.aes.Inference import read_manifest
    from ielts_judge.panda.Corpus import Corpus
    aes = read_manifest(paths["aes"])
    corpus = Corpus(paths["corpus"])
    _, experiment = prepare_experiment(train, test)
    if corpus.manifest["dataset_manifest_sha256"] != train.attrs["manifest_sha256"]:
        raise ValueError("Dataset and corpus releases differ")
    if (aes["provenance"]["dataset_manifest_sha256"] != train.attrs["manifest_sha256"]
            or aes["provenance"]["train"] != experiment["train"] or corpus.manifest["train"] != experiment["train"]):
        raise ValueError("Accepted AES/corpus train provenance differs from prepared data")
    files = {}
    for relative in ("code/offline_research/EvaluateModels.py", "code/shared/paths.yaml", "code/shared/config.yaml", "code/offline_research/requirements-evaluation.txt", "code/offline_research/evaluation/README.md",
                     "code/offline_research/tools/BuildEvaluationKaggle.py", "tests/TestModelEvaluation.py",
                     "code/shared/resources/ielts_criteria.yaml", "code/shared/resources/model_registry.yaml",
                     "code/offline_research/notebooks/ModelEvaluation.ipynb", "code/offline_research/notebooks/ModelEvaluationConfig.yaml", "code/offline_research/notebooks/ModelEvaluationModels.yaml",
                     config["inputs"]["aes_release"], config["inputs"]["corpus_release"]):
        files[relative] = ROOT / relative
    for folder in (ROOT / "code", paths["dataset"], paths["aes"], paths["corpus"]):
        for source in sorted(folder.rglob("*")):
            if source.is_relative_to(ROOT / "code") and set(source.relative_to(ROOT).parts).intersection({"artifacts", ".cache", "__pycache__", "tools", "notebooks", "legacy", "public_demo", "local_backup"}):
                continue
            if source.is_file() and "__pycache__" not in source.parts and (
                    not source.is_relative_to(ROOT / "code") or source.suffix == ".py"):
                files[source.relative_to(ROOT).as_posix()] = source
    hashes = {name: file_hash(path) for name, path in sorted(files.items())}
    bundle_id = inventory_id(hashes)
    output = Path(output_root or ROOT / "results/evaluation/kaggle") / (bundle_id[:12] + "-" + unique_name())
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / "model_evaluation_bundle.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Evaluation ZIP failed CRC verification")
    manifest = {"schema_version": "model-evaluation-bundle-v1", "bundle_id": bundle_id,
                "archive_name": archive_path.name, "archive_sha256": file_hash(archive_path), "files": hashes,
                "dataset_counts": {"train": len(train), "test": len(test)},
                "dataset_manifest_sha256": train.attrs["manifest_sha256"], "corpus_id": corpus.manifest["corpus_id"],
                "service_evaluation": "deferred"}
    (output / "ModelEvaluationBundleManifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "ModelEvaluation.ipynb").write_text(json.dumps(notebook_document(bundle_id), ensure_ascii=False, indent=1), encoding="utf-8")
    (output / "UPLOAD_INSTRUCTIONS.md").write_text(
        "# New Kaggle model evaluation\n\nCreate a NEW private Dataset named ielts-writing-model-evaluation.\n"
        "Upload model_evaluation_bundle.zip and ModelEvaluationBundleManifest.json together.\n"
        "Create a NEW notebook named IELTS Writing Model Evaluation and import the matching ModelEvaluation.ipynb.\n"
        "Attach only the new evaluation Dataset; enable GPU and Internet.\n"
        "Existing AES/PANDA/LLM Datasets and notebooks are not updated.\n"
        "See code/offline_research/evaluation/README.md for pilot, resume, plot selection and exports.\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--write-template", action="store_true", help="Create the new generic notebook once, without building assets")
    current_arguments = parser.parse_args()
    if current_arguments.write_template:
        with (ROOT / "code/offline_research/notebooks/ModelEvaluation.ipynb").open("x", encoding="utf-8") as stream:
            json.dump(notebook_document(), stream, ensure_ascii=False, indent=1)
        print(ROOT / "code/offline_research/notebooks/ModelEvaluation.ipynb")
    else:
        print(build_bundle(current_arguments.output_root))
