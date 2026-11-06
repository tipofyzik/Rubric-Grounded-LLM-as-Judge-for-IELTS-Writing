"""Sequential local-model experiments with append-only prediction segments."""
from typing import Any

from datetime import datetime, timezone
import gc
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import time
import warnings
from uuid import uuid4

import pandas as pd

from ielts_judge.data.Experiment import prepare_experiment
from ielts_judge.data.Preparation import file_hash, load_prepared
from ielts_judge.ModelRegistry import ModelRegistry
from offline_research.evaluation.Config import fingerprint
from offline_research.evaluation.Context import asset_paths


def unique_name() -> Any:
    """Unique name.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:10]


def write_new_json(path: Any, value: Any) -> None:
    """Write new json.

    Args:
        path (Any): Filesystem location to read, validate or write.
        value (Any): Value used by this operation.

    Returns:
        None: This operation returns no value.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace {path}")
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"Refusing to replace {path}")
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def environment() -> dict[str, Any]:
    """Environment.

    Returns:
        dict[str, Any]: Named result fields and metadata produced by this operation.
    """
    packages = {}
    for package in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "matplotlib", "torch",
                    "transformers", "huggingface-hub", "sentence-transformers", "langchain-core"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def initialize_run(root: Any, config: Any, output_root: Any, resume: Any=None) -> Any:
    """Initialize run.

    Args:
        root (Any): Project root used to resolve relative configuration paths.
        config (Any): Config used by this operation.
        output_root (Any): Output root used by this operation.
        resume (Any): Resume used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    root = Path(root).resolve()
    paths = asset_paths(root, config)
    train, pool, _ = load_prepared(paths["dataset"])
    selected, experiment = prepare_experiment(train, pool, config={current_key: current_value for current_key, current_value in config["evaluation"].items() if current_key != "repeats"})
    registry = ModelRegistry(root / config["model_registry"])
    sources = ["Prompts.py", "Scores.py", "Criteria.py", "Scoring.py", "ModelRegistry.py", "llm/Judge.py",
               "llm/Generation.py", "llm/JsonEosBlocker.py", "evaluation/Config.py", "evaluation/Context.py", "evaluation/Runtime.py"]
    source_hashes = {name: file_hash(root / ("code/offline_research" if name.startswith("evaluation/") else "code/shared/ielts_judge") / name) for name in sources}
    inference_config = {current_key: current_value for current_key, current_value in config.items() if current_key != "report"}
    identity = {"config": inference_config, "experiment": experiment,
                "aes_manifest": file_hash(paths["aes"] / "manifest.json"),
                "corpus_manifest": file_hash(paths["corpus"] / "manifest.json"),
                "registry": file_hash(root / config["model_registry"]), "source_hashes": source_hashes}
    current_environment = environment()
    identity["execution_environment"] = {"python": current_environment["python"], "packages": current_environment["packages"]}
    identity_hash = fingerprint(identity)
    if resume is not None:
        run = Path(resume).resolve()
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("run_fingerprint") != identity_hash:
            raise ValueError("Resume inputs/settings/source differ; create a new experiment")
        if json.loads((run / "experiment_manifest.json").read_text(encoding="utf-8")) != experiment:
            raise ValueError("Resume selection manifest differs")
        if file_hash(run / "reference.csv") != manifest["reference_sha256"]:
            raise ValueError("Saved reference labels were changed")
    else:
        run = Path(output_root).resolve() / identity_hash[:12] / unique_name()
        run.mkdir(parents=True, exist_ok=False)
        plan = {"models": config["models"], "setups": config["setups"], "repeats": config["evaluation"]["repeats"],
                "essay_ids": selected.essay_id.tolist(), "dataset_manifest_sha256": experiment["dataset_manifest_sha256"],
                "model_labels": {key: registry.require_enabled(key).label for key in config["models"]}}
        manifest = {"schema_version": "model-evaluation-run-v1", "run_fingerprint": identity_hash,
                    "identity": identity, "plan": plan, "environment": current_environment,
                    "service_evaluation": "deferred", "retry_policy": "one attempt per job; resume missing jobs only"}
        write_new_json(run / "run_config.json", config)
        write_new_json(run / "experiment_manifest.json", experiment)
        selected[["essay_id", *experiment["protocol"]["criteria"]]].assign(
            dataset_manifest=experiment["dataset_manifest_sha256"]).to_csv(run / "reference.csv", index=False)
        manifest["reference_sha256"] = file_hash(run / "reference.csv")
        bundle_manifest = root / "ModelEvaluationBundleManifest.json"
        manifest["input_bundle_id"] = json.loads(bundle_manifest.read_text(encoding="utf-8"))["bundle_id"] if bundle_manifest.exists() else None
        write_new_json(run / "run_manifest.json", manifest)
    return run, selected, experiment, manifest, paths


def prepare_run_context(run: Any, paths: Any, frame: Any, experiment: Any, manifest: Any, runtime: Any, preparer: Any=None) -> Any:
    """Prepare run context.

    Args:
        run (Any): Run used by this operation.
        paths (Any): Paths used by this operation.
        frame (Any): Frame used by this operation.
        experiment (Any): Experiment used by this operation.
        manifest (Any): Manifest used by this operation.
        runtime (Any): Runtime used by this operation.
        preparer (Any): Preparer used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    release_path = Path(run) / "context_release.json"
    if release_path.exists():
        contexts = read_run_context(run, manifest)
    else:
        if preparer is None:
            from offline_research.evaluation.Context import prepare_contexts
            preparer = prepare_contexts
        contexts, timing = preparer(paths, frame, experiment, device=runtime["context_device"],
                                   batch_size=runtime["aes_batch_size"], k=runtime["retrieval_k"])
        folder = Path(run) / "context" / unique_name()
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / "contexts.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for record in contexts:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        write_new_json(folder / "timings.json", timing)
        write_new_json(release_path, {"path": path.relative_to(run).as_posix(), "sha256": file_hash(path),
                                     "run_fingerprint": manifest["run_fingerprint"]})
    ids = [row["essay_id"] for row in contexts]
    expected = dict(zip(frame.essay_id, frame.essay_hash))
    if len(set(ids)) != len(ids) or set(ids) != set(expected) or any(current_record["essay_hash"] != expected[current_record["essay_id"]] for current_record in contexts):
        raise ValueError("Prepared context belongs to different evaluation essays")
    return contexts


def read_run_context(run: Any, manifest: Any) -> Any:
    """Read run context.

    Args:
        run (Any): Run used by this operation.
        manifest (Any): Manifest used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    run = Path(run).resolve()
    release = json.loads((run / "context_release.json").read_text(encoding="utf-8"))
    path = (run / release["path"]).resolve()
    if (not path.is_relative_to(run) or release["run_fingerprint"] != manifest["run_fingerprint"]
            or file_hash(path) != release["sha256"]):
        raise ValueError("Prepared context fingerprint/hash mismatch")
    rows = read_segment(path, tolerate_tail=False)
    ids = [row["essay_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(manifest["plan"]["essay_ids"]):
        raise ValueError("Prepared context IDs differ from the run plan")
    return rows


def read_segment(path: Any, *, tolerate_tail: bool=True) -> Any:
    """Read segment.

    Args:
        path (Any): Filesystem location to read, validate or write.
        tolerate_tail (bool): Tolerate tail used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    result = []
    with Path(path).open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b"\n") and tolerate_tail:
                warnings.warn(f"Uncommitted last line preserved and ignored: {Path(path).name}:{number}")
                break
            try:
                result.append(json.loads(line))
            except (UnicodeDecodeError, ValueError) as error:
                raise ValueError(f"Invalid completed journal record: {Path(path).name}:{number}") from error
    return result


def prediction_records(run: Any) -> list[Any]:
    """Prediction records.

    Args:
        run (Any): Run used by this operation.

    Returns:
        list[Any]: The computed result for the supplied inputs.
    """
    return [record for path in sorted((Path(run) / "predictions").glob("*.jsonl")) for record in read_segment(path)]


def job_key(record: Any) -> Any:
    """Job key.

    Args:
        record (Any): Record used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    return record["model_key"], record["setup"], record["repeat"], record["essay_id"]


def infer_models(root: Any, config: Any, run: Any, frame: Any, contexts: Any, manifest: Any, *, max_jobs: Any=None, judge_factory: Any=None) -> Any:
    """Infer models.

    Args:
        root (Any): Project root used to resolve relative configuration paths.
        config (Any): Config used by this operation.
        run (Any): Run used by this operation.
        frame (Any): Frame used by this operation.
        contexts (Any): Contexts used by this operation.
        manifest (Any): Manifest used by this operation.
        max_jobs (Any): Max jobs used by this operation.
        judge_factory (Any): Judge factory used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    from ielts_judge.llm.Judge import Judge
    from ielts_judge.Scoring import ScoringLimits, ScoringService
    from offline_research.evaluation.Context import PreparedAES, PreparedRetriever
    import torch

    class RecordingJudge(Judge):
        """Manage recording judge through the operations defined by this component."""
        def predict_messages(self, **keyword_arguments: Any) -> Any:
            """Generate a completion for validated chat messages within the shared deadline.

            Args:
                keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            self.last_generation = {}
            self.last_error_type = None
            gpu = self.model.device.type == "cuda"
            if gpu:
                torch.cuda.synchronize(self.model.device)
                torch.cuda.reset_peak_memory_stats(self.model.device)
            tick = time.perf_counter()
            try:
                result = super().predict_messages(**keyword_arguments)
                if gpu:
                    torch.cuda.synchronize(self.model.device)
                    result.update(gpu_work_seconds=time.perf_counter() - tick,
                                  gpu_peak_allocated_mib=torch.cuda.max_memory_allocated(self.model.device) / 2**20)
                self.last_generation = result
                return result
            except Exception as error:
                self.last_error_type = type(error).__name__
                raise

    if max_jobs is not None and (type(max_jobs) is not int or max_jobs < 1):
        raise ValueError("max_jobs must be a positive integer")
    registry = ModelRegistry(Path(root) / config["model_registry"])
    existing = prediction_records(run)
    done = {job_key(current_record) for current_record in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate completed jobs in prediction journals")
    expected = {(model, setup, repeat, essay_id) for model in config["models"] for setup in config["setups"]
                for repeat in range(config["evaluation"]["repeats"]) for essay_id in frame.essay_id}
    if not done.issubset(expected) or any(current_record.get("dataset_manifest") != manifest["plan"]["dataset_manifest_sha256"]
        or current_record.get("metadata", {}).get("versions", {}).get("evaluation") != manifest["run_fingerprint"] for current_record in existing):
        raise ValueError("Saved prediction journal belongs to a different experiment")
    if done == expected:
        return 0
    context = {row["essay_id"]: row for row in contexts}
    rows = {row["essay_id"]: row for row in frame[["essay_id", "task_prompt", "essay"]].to_dict("records")}
    folder = Path(run) / "predictions"
    folder.mkdir(exist_ok=True)
    session = unique_name()
    env = environment()
    env["cuda"] = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda,
                   "devices": [torch.cuda.get_device_name(current_index) for current_index in range(torch.cuda.device_count())],
                   "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    for previous in Path(run).glob("inference_environment_*.json"):
        previous_environment = json.loads(previous.read_text(encoding="utf-8"))
        if existing and (previous_environment["cuda"] != env["cuda"] or previous_environment["packages"] != env["packages"]
                         or previous_environment["python"] != env["python"]):
            raise ValueError("Inference hardware/packages changed; start a separate run instead of mixing results")
    write_new_json(Path(run) / f"inference_environment_{session}.json", env)
    completed = 0
    with (folder / f"{session}.jsonl").open("x", encoding="utf-8") as journal:
        for model_key in config["models"]:
            pending = [(model_key, setup, repeat, essay_id) for repeat in range(config["evaluation"]["repeats"])
                       for setup in config["setups"] for essay_id in frame.essay_id if (model_key, setup, repeat, essay_id) not in done]
            if not pending:
                continue
            if judge_factory is None and config["runtime"]["judge_device"] == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; enable the Kaggle GPU before loading judge weights")
            load_started = time.perf_counter()
            if judge_factory is None:
                judge = RecordingJudge.from_pretrained(model_key, device=config["runtime"]["judge_device"], registry=registry,
                                                       token=os.environ.get("HF_TOKEN"))
            else:
                judge = judge_factory(model_key, registry)
            try:
                loads = Path(run) / "model_loads"
                loads.mkdir(exist_ok=True)
                write_new_json(loads / f"{session}-{model_key}.json", {
                    "model_key": model_key, "session": session, "load_seconds": time.perf_counter() - load_started,
                    "requested_device": config["runtime"]["judge_device"],
                    "scope": "model/tokenizer startup, including any uncached download; separate from per-essay judge time",
                    "cuda_allocated_mib": torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else None,
                })
                for _, setup, repeat, essay_id in pending:
                    if max_jobs is not None and completed >= max_jobs:
                        return completed
                    torch.manual_seed(config["evaluation"]["seed"] + repeat)
                    prepared = context[essay_id]
                    judge.last_generation = {}
                    judge.last_error_type = None
                    service = ScoringService(judge, aes=PreparedAES(prepared), retriever=PreparedRetriever(prepared), setup=setup,
                        limits=ScoringLimits(timeout_seconds=config["runtime"]["timeout_seconds"], max_input_chars=config["runtime"]["max_input_chars"]),
                        versions={"evaluation": manifest["run_fingerprint"], "context": "precomputed real AES/BGE; offline train retrieval"})
                    result = service.score(model_key=model_key, essay_id=essay_id,
                                           task_prompt=rows[essay_id]["task_prompt"], essay=rows[essay_id]["essay"])
                    generation = getattr(judge, "last_generation", {})
                    if generation:
                        result["metadata"].setdefault("generation", {}).update({key: generation[key] for key in (
                            "input_tokens", "generated_tokens", "max_new_tokens", "stop_reason", "gpu_work_seconds", "gpu_peak_allocated_mib")
                            if key in generation})
                        result["generation_diagnostics"] = {key: value for key, value in generation.items() if key != "raw_text"}
                    result.update(essay_id=essay_id, setup=setup, repeat=repeat,
                                  dataset_manifest=manifest["plan"]["dataset_manifest_sha256"],
                                  raw_completion=getattr(judge, "last_generation", {}).get("raw_text"),
                                  judge_exception_type=getattr(judge, "last_error_type", None))
                    journal.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    completed += 1
                    if completed % 25 == 0:
                        print(f"Saved {completed} new jobs; latest {model_key}/{setup}/repeat={repeat}", flush=True)
                    del service
                    if result.get("error", {}).get("code") == "judge_failure":
                        raise RuntimeError("Judge runtime failed; result preserved. Inspect the journal before resuming.")
            finally:
                if hasattr(judge, "model"):
                    del judge.model
                if hasattr(judge, "tokenizer"):
                    del judge.tokenizer
                del judge
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return completed


def report_run(run: Any, contexts: Any, manifest: Any, options: Any) -> Any:
    """Report run.

    Args:
        run (Any): Run used by this operation.
        contexts (Any): Contexts used by this operation.
        manifest (Any): Manifest used by this operation.
        options (Any): Options used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    from offline_research.evaluation.Report import write_report
    records = [{"model_key": "aes", "setup": "aes", "repeat": 0, "essay_id": row["essay_id"],
                "dataset_manifest": manifest["plan"]["dataset_manifest_sha256"], "status": "ok", "scores": row["aes"]["scores"]}
               for row in contexts]
    records.extend(prediction_records(run))
    if file_hash(Path(run) / "reference.csv") != manifest["reference_sha256"]:
        raise ValueError("Saved reference labels were changed")
    truth = pd.read_csv(Path(run) / "reference.csv")
    loads = [json.loads(candidate_path.read_text(encoding="utf-8")) for candidate_path in sorted((Path(run) / "model_loads").glob("*.json"))]
    return write_report(records, truth, manifest["plan"], Path(run) / "reports" / unique_name(), options, model_loads=loads)


def export_run(run: Any) -> Any:
    """Export run.

    Args:
        run (Any): Run used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    from offline_research.evaluation.Results import export_results
    return export_results(run)
