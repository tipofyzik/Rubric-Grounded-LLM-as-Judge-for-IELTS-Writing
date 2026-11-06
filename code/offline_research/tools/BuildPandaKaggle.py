"""Package PANDA generation code, verified data and a hash-pinned Kaggle notebook."""
from typing import Any

import argparse
import hashlib
import json
import inspect
from pathlib import Path
import sys
import zipfile

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]


def prepare_project(
    input_root: Any,
    working_root: Any,
    bundle_path: Any=None,
    expected_sha256: Any=None,
    *,
    expected_files: Any=None,
) -> Any:
    """Prepare project.

    Args:
        input_root (Any): Input root used by this operation.
        working_root (Any): Working root used by this operation.
        bundle_path (Any): Bundle path used by this operation.
        expected_sha256 (Any): Expected sha256 used by this operation.
        expected_files (Any): Expected files used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    from pathlib import Path
    import hashlib
    import shutil
    import uuid
    import zipfile

    input_root = Path(input_root)
    candidates = [Path(bundle_path)] if bundle_path else []
    if not bundle_path:
        for path in input_root.rglob("*.zip"):
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    if "code/offline_research/ManageCorpus.py" in archive.namelist():
                        candidates.append(path)
        if not candidates:
            candidates = sorted(
                {
                    path.parent.parent.parent
                    for path in input_root.rglob("ManageCorpus.py")
                    if path.parent.name == "offline_research"
                }
            )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one PANDA runtime; found {candidates}. Use Add Input or set bundle_path."
        )
    source = candidates[0]
    if (
        source.is_file()
        and expected_sha256
        and hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256
    ):
        raise ValueError("Runtime ZIP SHA256 mismatch")
    if source.is_dir() and not expected_files:
        raise ValueError("Unpacked input requires expected_files hashes")
    project = Path(working_root) / ("ielts-panda-" + uuid.uuid4().hex[:10])
    project.mkdir(parents=True, exist_ok=False)
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                if (
                    not (project / member.filename)
                    .resolve()
                    .is_relative_to(project.resolve())
                ):
                    raise ValueError("Unsafe archive path")
            archive.extractall(project)
    else:
        for name in expected_files or {}:
            original, target = (source / name).resolve(), (project / name).resolve()
            if not original.is_relative_to(
                source.resolve()
            ) or not target.is_relative_to(project.resolve()):
                raise ValueError("Unsafe input path")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    for name, checksum in (expected_files or {}).items():
        target = (project / name).resolve()
        if not target.is_relative_to(project.resolve()) or not target.is_file():
            raise ValueError(f"Missing/unsafe runtime file: {name}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Runtime file SHA256 mismatch: {name}")
    if not (project / "code/offline_research/ManageCorpus.py").is_file():
        raise ValueError("Missing code/offline_research/ManageCorpus.py")
    return project


def notebook_token(use_hf_token: bool, secret_name: Any) -> Any:
    """Read an explicitly enabled Kaggle secret without printing or persisting it.

    Args:
        use_hf_token (bool): Use hf token used by this operation.
        secret_name (Any): Secret name used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if not use_hf_token:
        return None
    try:
        import importlib

        token = (
            importlib.import_module("kaggle_secrets")
            .UserSecretsClient()
            .get_secret(secret_name)
        )
    except Exception:
        # Secret backend errors may contain response bodies; don't expose them in notebook output.
        raise RuntimeError(
            "Enable the selected Hugging Face token secret in Kaggle Secrets, then rerun this cell."
        ) from None
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError("The selected Kaggle HF token secret is empty")
    return token.strip()


def configure_generation(
    project_root: Any, model_id: Any, model_revision: Any, run_name: Any, token: Any, prompt_variant: Any=None, batch_size: int=None
) -> Any:
    """Write a separate profile with a pinned revision; keep credentials out of artifacts.

    Args:
        project_root (Any): Selected source checkout used for configuration and artifacts.
        model_id (Any): Model id used by this operation.
        model_revision (Any): Model revision used by this operation.
        run_name (Any): Run name used by this operation.
        token (Any): Token used by this operation.
        prompt_variant (Any): Prompt variant used by this operation.
        batch_size (int): Batch size used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    import json
    import re
    from pathlib import Path
    import yaml

    if not isinstance(run_name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name
    ):
        raise ValueError(
            "RUN_NAME must contain 1-64 letters, digits, underscores or hyphens"
        )
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("MODEL_ID must be a Hugging Face repository ID")
    if model_revision is not None and (
        not isinstance(model_revision, str) or not model_revision.strip()
    ):
        raise ValueError("MODEL_REVISION must be None, a commit, tag or branch")
    project_root = Path(project_root)
    profile = yaml.safe_load(
        (project_root / "code/offline_research/resources/panda_generation.yaml").read_text(encoding="utf-8")
    )
    generator = profile["generator"]
    if prompt_variant is not None:
        generator["prompt_variant"] = prompt_variant
    if batch_size is not None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("BATCH_SIZE must be a positive integer or None to use panda_generation.yaml")
        generator["batch_size"] = batch_size
    selected_revision = model_revision
    saved = project_root / run_name / "generation_manifest.json"
    if not saved.exists():
        saved = project_root / run_name / "comparison_manifest.json"
    if selected_revision is None and saved.exists():
        previous = json.loads(saved.read_text(encoding="utf-8"))["generator"]
        if previous["model_id"] == model_id:
            selected_revision = previous["revision"]
    if selected_revision is None and generator["model_id"] == model_id:
        selected_revision = generator["revision"]
    if selected_revision is None or not re.fullmatch(
        r"[0-9a-f]{40}", selected_revision
    ):
        from huggingface_hub import HfApi

        selected_revision = (
            HfApi(token=token or False)
            .model_info(model_id, revision=selected_revision or "main")
            .sha
        )
    if not isinstance(selected_revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}", selected_revision
    ):
        raise ValueError("Hugging Face did not return an immutable model commit")
    generator.update(model_id=model_id, revision=selected_revision)
    destination = project_root / (run_name + ".profile.yaml")
    destination.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return destination


def notebook(bundle_hash: Any, hashes: Any) -> dict[str, Any]:
    """Notebook.

    Args:
        bundle_hash (Any): Bundle hash used by this operation.
        hashes (Any): Hashes used by this operation.

    Returns:
        dict[str, Any]: Named result fields and metadata produced by this operation.
    """
    bootstrap = (
        "from typing import Any\nfrom pathlib import Path\nimport hashlib, json, os, subprocess, sys, uuid, zipfile\n\n"
        + inspect.getsource(prepare_project)
    )
    cells = []

    def add(kind: Any, source: Any) -> None:
        """Add.

        Args:
            kind (Any): Kind used by this operation.
            source (Any): Source used by this operation.

        Returns:
            None: This operation returns no value.
        """
        cell = {
            "cell_type": kind,
            "id": f"panda-generation-{len(cells)}",
            "metadata": {},
            "source": source,
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add(
        "markdown",
        "# PANDA: batched insight generation\nAttach the matching PANDA runtime ZIP using Add Input, enable Internet and a GPU. Default: adjacent_text v5, 40 new essay attempts, batches of 12. Set BATCH_SIZE below, or None to use generator.batch_size from panda_generation.yaml. Batch size is part of the saved profile: use a new RUN_NAME when changing it. LIMIT counts essays, not batches; None processes all remaining train essays. Sequential weight loading is enabled. In generate mode rerunning advances to the next unfinished essays.\n",
    )
    add("code", bootstrap)
    add(
        "code",
        "bundle_path = None  # Set explicitly if more than one runtime is attached.\n"
        + "expected_sha256 = "
        + repr(bundle_hash)
        + "\nexpected_files = "
        + repr(hashes)
        + "\n"
        + 'project_root = prepare_project("/kaggle/input", "/kaggle/working", bundle_path, expected_sha256, expected_files=expected_files)\n',
    )
    add(
        "code",
        """# Keep Kaggle's CUDA torch; isolate pinned inference/data dependencies without ensurepip.
import venv
runtime_dir = project_root / ".venv"
venv.EnvBuilder(with_pip=False, system_site_packages=True).create(runtime_dir)
python = runtime_dir / "bin/python"
subprocess.run([sys.executable, "-m", "pip", "--python", str(python), "install", "pip==25.3"], check=True)
subprocess.run([str(python), "-m", "pip", "install", "numpy==2.4.6", "pandas==3.0.1", "pyarrow==23.0.1",
    "PyYAML==6.0.3", "requests==2.32.5", "transformers==5.2.0", "huggingface-hub==1.6.0",
    "sentence-transformers==5.2.3", "accelerate==1.13.0"], check=True)
subprocess.run([str(python), "-c", "import torch; assert torch.cuda.is_available(), 'Select a Kaggle GPU'; print(torch.__version__, torch.cuda.get_device_name(0))"], check=True)
""",
    )
    add(
        "code",
        """# Choose the INSIGHT generator here, before resume/generation.
from typing import Any
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = None  # None: saved/default commit; another model resolves main once.
RUN_MODE = "generate"  # "generate": one selected prompt; "compare": the previous two-prompt A/B.
PROMPT_VARIANT = "adjacent_text"  # Explicit source band and neighbors; original_text / legacy_text / simple_json remain available.
RUN_NAME = "panda_adjacent_bands_v5_batch12"  # New name when changing mode/model/profile/batch size.
BATCH_SIZE = 12  # Positive integer; None uses generator.batch_size from panda_generation.yaml.
USE_HF_TOKEN = True  # Enable the HF_TOKEN secret in Kaggle Secrets.
HF_TOKEN_SECRET = "HF_TOKEN"  # Secret label only; never paste the token into a cell.

"""
        + inspect.getsource(notebook_token)
        + "\n\n"
        + inspect.getsource(configure_generation)
        + """
hf_token = notebook_token(USE_HF_TOKEN, HF_TOKEN_SECRET)
import re
if not isinstance(RUN_NAME, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", RUN_NAME):
    raise ValueError("Use a simple RUN_NAME containing letters, digits, underscores or hyphens")
run_dir = project_root / RUN_NAME
""",
    )
    add(
        "code",
        """# Optional: attach the previous output ZIP and set RESUME_ZIP to continue after a Kaggle restart.
RESUME_ZIP = None
if RESUME_ZIP:
    if run_dir.exists():
        raise ValueError("Resume into a fresh project, before generating new responses")
    with zipfile.ZipFile(RESUME_ZIP) as archive:
        for member in archive.infolist():
            target = (project_root / member.filename).resolve()
            if not target.is_relative_to(run_dir.resolve()):
                raise ValueError("Unsafe resume archive path")
        archive.extractall(project_root)
    manifest_name = "comparison_manifest.json" if RUN_MODE == "compare" else "generation_manifest.json"
    if not (run_dir / manifest_name).is_file():
        raise ValueError("Resume ZIP lacks the selected run manifest")
""",
    )
    add(
        "code",
        """# compare always stays within the frozen 40 IDs; rerunning never advances to another 40.
# Default adjacent_text v5 before-answering pilot: 40 new attempts; run once, then export and review.
# generate with LIMIT=40 caps NEW attempts per invocation and advances on each rerun.
# RETRY_FAILED explicitly retries failed IDs with the SAME profile, never changes prompts.
LIMIT = 40  # New attempts per invocation; compare applies the limit separately per prompt.
RETRY_FAILED = False
if RUN_MODE not in ("compare", "generate"):
    raise ValueError('RUN_MODE must be "compare" or "generate"')
profile_path = configure_generation(project_root, MODEL_ID, MODEL_REVISION, RUN_NAME, hf_token, PROMPT_VARIANT, BATCH_SIZE)
command = [str(python), "-B", "code/offline_research/ManageCorpus.py", RUN_MODE, "--output", str(run_dir), "--profile", str(profile_path)]
if LIMIT is not None:
    command += ["--limit", str(LIMIT)]
if RETRY_FAILED:
    command += ["--retry-failed"]
generation_env = os.environ.copy()
if hf_token is not None:
    generation_env["HF_TOKEN"] = hf_token
else:
    generation_env.pop("HF_TOKEN", None)
generation_env["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
generation_env["PYTHONUNBUFFERED"] = "1"
subprocess.run(command, cwd=project_root, env=generation_env, check=True)
report_name = "comparison_report.json" if RUN_MODE == "compare" else "generation_report.json"
print((run_dir / report_name).read_text())
""",
    )
    add(
        "code",
        """# Run after the pilot, full run, or an interrupted run to download completed progress.
if not any((run_dir / name).is_file() for name in ("generation_manifest.json", "comparison_manifest.json")):
    raise ValueError("No generation results to export")
output_zip = Path("/kaggle/working/panda_generation_results.zip")
with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.suffix != ".tmp":
            archive.write(path, path.relative_to(project_root))
print("Download:", output_zip)
print("SHA256:", hashlib.sha256(output_zip.read_bytes()).hexdigest())
""",
    )
    add(
        "markdown",
        "Download panda_generation_results.zip and send it back. Each batch is generated together, then each essay is validated and saved separately by stable ID. Failed results retain completion, error and any recognized insights. An interrupted batch may need regeneration; completed earlier batches remain resumable. CUDA/OOM errors stop the run; export progress and choose a smaller batch under a new RUN_NAME if needed. For throughput use processing_seconds_this_invocation and seconds_per_attempt_this_invocation from the report; per-record seconds includes shared batch latency and must not be summed as total runtime. The first batch includes model loading. Greedy decoding, prompt and token budgets are unchanged. Text results remain pilot/comparison-only.\n",
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(output: Any, prepared: Any=None) -> Any:
    """Build.

    Args:
        output (Any): Output used by this operation.
        prepared (Any): Prepared used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    from ielts_judge.data.Preparation import file_hash, load_prepared
    from ielts_judge.Paths import PATHS

    release = Path(prepared or PATHS["datasets"]["prepared"])
    _, _, manifest = load_prepared(release)
    files = {
        relative: ROOT / relative
        for relative in (
            "code/offline_research/ManageCorpus.py",
            "code/offline_research/config.yaml",
            "code/shared/paths.yaml", "code/shared/config.yaml",
            "code/shared/resources/ielts_criteria.yaml",
            "code/shared/resources/model_registry.yaml",
            "code/offline_research/resources/panda_generation.yaml",
            "code/offline_research/resources/panda_comparison.json",
        )
    }
    for path in (ROOT / "code").rglob("*.py"):
        if set(path.relative_to(ROOT).parts).intersection({"artifacts", ".cache", "__pycache__", "tools", "notebooks", "legacy", "public_demo", "local_backup"}):
            continue
        files[path.relative_to(ROOT).as_posix()] = path
    target = "data/datasets/processed/ielts-v3/"
    for name in ("manifest.json", *manifest["artifacts"]):
        files[target + name] = release / name
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "panda_generation_bundle.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name, source in sorted(files.items()):
            archive.write(source, name)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("PANDA runtime ZIP failed CRC verification")
    hashes = {name: file_hash(source) for name, source in sorted(files.items())}
    checksum = file_hash(archive_path)
    (output / "PandaGeneration.ipynb").write_text(
        json.dumps(notebook(checksum, hashes), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "bundle_manifest.json").write_text(
        json.dumps({"archive_sha256": checksum, "files": hashes}, indent=2),
        encoding="utf-8",
    )
    return archive_path


def main(command_arguments: Any=None) -> Any:
    """Parse command options and execute this component's command-line operation.

    Args:
        command_arguments (Any): Explicit CLI arguments; None reads the process argument list.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "panda_generation" / "kaggle",
        help="Destination directory (default: %(default)s)",
    )
    parser.add_argument("--prepared", type=Path)
    current_arguments = parser.parse_args(command_arguments)
    archive_path = build(current_arguments.output, current_arguments.prepared)
    print(archive_path)
    return archive_path


if __name__ == "__main__":
    main()
