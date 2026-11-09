"""Offline PANDA generation, corpus building and Supabase RPC verification."""
from typing import Any

from pathlib import Path
import sys

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

from ielts_judge.data.Experiment import prepare_experiment
from ielts_judge.data.Preparation import load_prepared, text_hash
from ielts_judge.Paths import PATHS
from ielts_judge.panda.Artifacts import canonical, read_json, write_json
from ielts_judge.panda.Corpus import Corpus, Encoder, embedding_profile
from offline_research.panda.Corpus import build_corpus
from offline_research.panda.Comparison import run_comparison
from offline_research.panda.Contracts import generation_profile, run_manifest
from offline_research.panda.Generation import HFInsightGenerator, run_insights


def data(prepared: Any) -> Any:
    """Load the reference train split and its experiment manifest.

    Args:
        prepared (Any): Directory of the prepared dataset release.

    Returns:
        Any: (Train DataFrame, experiment manifest).
    """
    train, test, _ = load_prepared(prepared)
    _, experiment = prepare_experiment(train, test)
    return train, experiment


def rpc_from_env(current_arguments: Any) -> Any:
    """Create a PANDA RPC client using process environment settings.

    Args:
        current_arguments (Any): Parsed options naming the URL and key environment variables.

    Returns:
        Any: SupabaseRPC configured for the supplied credentials.

    Raises:
        ValueError: A required variable is missing or invalid.
    """
    from ielts_judge.Database import SupabaseRPC
    values = [os.environ.get(name) for name in (current_arguments.url_env, current_arguments.key_env)]
    if not all(values):
        raise ValueError(f"Set {current_arguments.url_env} and {current_arguments.key_env} in this process; do not put credentials in arguments")
    return SupabaseRPC(values[0], values[1])


def verify_rpc(corpus: Any, rpc: Any, *, query_count: int=5) -> dict[str, Any]:
    """Compare remote matches with local cosine search over the same saved vectors.

    Args:
        corpus (Any): Verified local corpus used as the comparison reference.
        rpc (Any): Client authorized to search the same corpus release.
        query_count (int): Number of saved queries to select deterministically.

    Returns:
        dict[str, Any]: Corpus ID, configured reference count and evidence for each query.

    Raises:
        ValueError: Remote neighbors, similarity or reference data differ from the local corpus.
    """
    from ielts_judge.Database import ReferenceRetriever, retrieval_k
    reference_count = retrieval_k()
    records = [record for _, record in corpus.records()]
    matrix = np.asarray([row["embedding"] for row in records], dtype=np.float32)
    profile = corpus.manifest["embedding"]
    class SavedEncoder:
        """Expose the selected saved embedding through the query encoder interface."""
        def __init__(self) -> None:
            """Keep the corpus profile and wait for a query row to be selected."""
            self.profile = profile
            self.row = None
        def encode(self, texts: Any) -> Any:
            """Return the selected essay's stored vector and token count.

            Args:
                texts (Any): One-element list containing the selected essay text.

            Returns:
                Any: (Single-row embedding array, one-element token length list).

            Raises:
                ValueError: The input differs from the selected essay.
            """
            if self.row is None or texts != [self.row["essay"]]:
                raise ValueError("Verification requires the selected saved essay")
            return np.array([self.row["embedding"]], dtype=np.float32), [self.row["original_tokens"]]
    encoder = SavedEncoder()
    retriever = ReferenceRetriever(rpc, corpus.manifest, encoder)
    evidence = []
    for row in sorted(records, key=lambda item: hashlib.sha256(item["essay_id"].encode()).hexdigest())[:query_count]:
        encoder.row = row
        result = retriever.retrieve(row["essay"], essay_id=row["essay_id"], reference_count=reference_count)
        query = np.asarray(row["embedding"], dtype=np.float32)
        similarities = (matrix @ query) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))
        eligible = [current_index for current_index, item in enumerate(records) if item["essay_id"] != row["essay_id"] and item["essay_hash"] != text_hash(row["essay"])]
        eligible.sort(key=lambda current_index: (-float(similarities[current_index]), records[current_index]["essay_id"]))
        returned = result["references"]
        if len(returned) != min(reference_count, len(eligible)):
            raise ValueError("RPC did not return the expected number of neighbors")
        by_id = {record["essay_id"]: current_index for current_index, record in enumerate(records)}
        cutoff = float(similarities[eligible[len(returned)-1]]) if returned else -1
        for reference, similarity in zip(returned, result["similarities"]):
            index = by_id.get(reference.essay_id)
            if index is None or abs(similarity - float(similarities[index])) > 1e-5 or similarity < cutoff - 1e-5:
                raise ValueError("RPC neighbors differ from exact local cosine search")
            expected = records[index]
            expected_insights = expected["insights"] if corpus.manifest["schema_version"] == "panda-corpus-v2" else canonical({key: {current_field: current_value[current_field] for current_field in ("evidence","why_not_higher","why_not_lower")} for key,current_value in expected["criteria"].items()})
            if reference.task_prompt != expected["task_prompt"] or reference.essay != expected["essay"] or reference.scores != {key: current_value["score"] for key,current_value in expected["criteria"].items()} or reference.insights != expected_insights:
                raise ValueError("RPC labels/insights do not match the saved reference ID")
        evidence.append({"query_id": row["essay_id"], "reference_ids": [current_record.essay_id for current_record in returned], "similarities": result["similarities"]})
    return {"corpus_id": corpus.manifest["corpus_id"], "k": reference_count, "queries": evidence, "status": "passed",
            "scope": "real RPC vs saved vectors; fresh-text encoder smoke is a separate query command"}


def main(command_arguments: Any=None) -> Any:
    """Build, inspect or query a PANDA corpus from the command line.

    The ingest command writes the corpus to Supabase. Query and verify-rpc
    read the remote corpus and save their results locally.

    Args:
        command_arguments (Any): Command-line argument strings, or None to use sys.argv.

    Returns:
        Any: Result dictionary for the selected command, also printed as JSON.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "generate", "compare", "build"):
        command = commands.add_parser(name)
        command.add_argument("--prepared", type=Path, default=Path(PATHS["datasets"]["prepared"]))
        command.add_argument("--profile", type=Path, default=Path(__file__).parent / "resources/panda_generation.yaml")
        command.add_argument("--output", type=Path, required=True)
        if name in ("generate", "compare"):
            command.add_argument("--limit", type=int)
            command.add_argument("--retry-failed", action="store_true")
        if name == "compare":
            command.add_argument("--selection", type=Path, default=Path(__file__).parent / "resources/panda_comparison.json")
        if name == "build":
            command.add_argument("--insights", type=Path, required=True)
            command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for name in ("inspect", "ingest", "verify-rpc", "query"):
        command = commands.add_parser(name)
        command.add_argument("--corpus", type=Path, required=True)
        if name != "inspect":
            command.add_argument("--url-env", default="SUPABASE_URL")
            command.add_argument("--key-env", default="SUPABASE_SECRET_KEY" if name == "ingest" else "SUPABASE_PUBLISHABLE_KEY")
        if name in ("verify-rpc", "query"):
            command.add_argument("--output", type=Path, required=True)
        if name == "query":
            command.add_argument("--essay-file", type=Path, required=True)
    current_arguments = parser.parse_args(command_arguments)
    if current_arguments.command in ("audit", "generate", "compare", "build"):
        config = yaml.safe_load(current_arguments.profile.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or set(config) != {"generator", "embedding"}:
            raise ValueError("PANDA profile needs generator and embedding sections")
        profile, embedding = generation_profile(config["generator"]), embedding_profile(config["embedding"])
        train, experiment = data(current_arguments.prepared)
        if current_arguments.command == "audit":
            current_arguments.output.mkdir(parents=True, exist_ok=False)
            write_json(current_arguments.output / "experiment_manifest.json", experiment)
            write_json(current_arguments.output / "generation_manifest.json", run_manifest(train, experiment, profile))
            result = {"count": len(train), "sources": train.dataset_source.value_counts().to_dict(), "embedding": embedding}
            write_json(current_arguments.output / "audit.json", result)
        elif current_arguments.command in ("generate", "compare"):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Generation requires a CUDA accelerator; use the supplied Kaggle notebook")
            write_json(current_arguments.output / "experiment_manifest.json", experiment)
            write_json(current_arguments.output / "environment.json", {"python": sys.version,
                "packages": {name: importlib.metadata.version(name) for name in ("torch","transformers","sentence-transformers","huggingface-hub","numpy","pandas")},
                "cuda": torch.version.cuda, "gpu": [torch.cuda.get_device_name(current_index) for current_index in range(torch.cuda.device_count())]})
            generator = HFInsightGenerator(profile)
            if current_arguments.command == "compare":
                result = run_comparison(train, experiment, current_arguments.output, profile, read_json(current_arguments.selection),
                    lambda records, variant: generator.generate_batch(records, prompt_variant=variant),
                    limit=current_arguments.limit, retry_failed=current_arguments.retry_failed)
            else:
                result = run_insights(train, experiment, current_arguments.output, profile, generator.generate_batch,
                                      limit=current_arguments.limit, retry_failed=current_arguments.retry_failed)
        else:
            result = build_corpus(train, experiment, current_arguments.insights, current_arguments.output, embedding, Encoder(embedding, device=current_arguments.device))
    else:
        corpus = Corpus(current_arguments.corpus)
        if current_arguments.command == "inspect":
            result = corpus.manifest
        else:
            from ielts_judge.Database import ReferenceRetriever, import_corpus
            rpc = rpc_from_env(current_arguments)
            if current_arguments.command == "ingest":
                result = import_corpus(corpus, rpc)
            elif current_arguments.command == "verify-rpc":
                result = verify_rpc(corpus, rpc)
                write_json(current_arguments.output, result)
            else:
                retriever = ReferenceRetriever(rpc, corpus.manifest, Encoder(corpus.manifest["embedding"]))
                found = retriever.retrieve(current_arguments.essay_file.read_text(encoding="utf-8"))
                result = {**found, "references": [current_record.__dict__ for current_record in found["references"]]}
                write_json(current_arguments.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
