"""Stage 4 offline generation, corpus building and Supabase RPC verification."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

from ielts_judge.data.experiment import prepare_experiment
from ielts_judge.data.preparation import load_prepared, text_hash
from ielts_judge.paths import PATHS
from ielts_judge.panda.artifacts import canonical, read_json, write_json
from ielts_judge.panda.corpus import Corpus, Encoder, build_corpus, embedding_profile
from ielts_judge.panda.comparison import run_comparison
from ielts_judge.panda.generation import HFInsightGenerator, generation_profile, run_insights, run_manifest


def data(prepared):
    train, test, _ = load_prepared(prepared)
    _, experiment = prepare_experiment(train, test)
    return train, experiment


def rpc_from_env(args):
    from ielts_judge.database import SupabaseRPC
    values = [os.environ.get(name) for name in (args.url_env, args.key_env)]
    if not all(values):
        raise ValueError(f"Set {args.url_env} and {args.key_env} in this process; do not put credentials in arguments")
    return SupabaseRPC(values[0], values[1])


def verify_rpc(corpus, rpc, *, query_count=5):
    """Compare true RPC and local cosine on identical saved vectors, with numerical tie tolerance."""
    from ielts_judge.database import ReferenceRetriever, retrieval_k
    k = retrieval_k()
    records = [record for _, record in corpus.records()]
    matrix = np.asarray([row["embedding"] for row in records], dtype=np.float32)
    profile = corpus.manifest["embedding"]
    class SavedEncoder:
        def __init__(self):
            self.profile = profile
            self.row = None
        def encode(self, texts):
            if self.row is None or texts != [self.row["essay"]]:
                raise ValueError("Verification requires the selected saved essay")
            return np.array([self.row["embedding"]], dtype=np.float32), [self.row["original_tokens"]]
    encoder = SavedEncoder()
    retriever = ReferenceRetriever(rpc, corpus.manifest, encoder)
    evidence = []
    for row in sorted(records, key=lambda item: hashlib.sha256(item["essay_id"].encode()).hexdigest())[:query_count]:
        encoder.row = row
        result = retriever.retrieve(row["essay"], essay_id=row["essay_id"], k=k)
        query = np.asarray(row["embedding"], dtype=np.float32)
        similarities = (matrix @ query) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))
        eligible = [i for i, item in enumerate(records) if item["essay_id"] != row["essay_id"] and item["essay_hash"] != text_hash(row["essay"])]
        eligible.sort(key=lambda i: (-float(similarities[i]), records[i]["essay_id"]))
        returned = result["references"]
        if len(returned) != min(k, len(eligible)):
            raise ValueError("RPC did not return the expected number of neighbors")
        by_id = {record["essay_id"]: i for i, record in enumerate(records)}
        cutoff = float(similarities[eligible[len(returned)-1]]) if returned else -1
        for reference, similarity in zip(returned, result["similarities"]):
            index = by_id.get(reference.essay_id)
            if index is None or abs(similarity - float(similarities[index])) > 1e-5 or similarity < cutoff - 1e-5:
                raise ValueError("RPC neighbors differ from exact local cosine search")
            expected = records[index]
            expected_insights = expected["insights"] if corpus.manifest["schema_version"] == "panda-corpus-v2" else canonical({key: {f: v[f] for f in ("evidence","why_not_higher","why_not_lower")} for key,v in expected["criteria"].items()})
            if reference.task_prompt != expected["task_prompt"] or reference.essay != expected["essay"] or reference.scores != {key: v["score"] for key,v in expected["criteria"].items()} or reference.insights != expected_insights:
                raise ValueError("RPC labels/insights do not match the saved reference ID")
        evidence.append({"query_id": row["essay_id"], "reference_ids": [r.essay_id for r in returned], "similarities": result["similarities"]})
    return {"corpus_id": corpus.manifest["corpus_id"], "k": k, "queries": evidence, "status": "passed",
            "scope": "real RPC vs saved vectors; fresh-text encoder smoke is a separate query command"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "generate", "compare", "build"):
        command = commands.add_parser(name)
        command.add_argument("--prepared", type=Path, default=Path(PATHS["datasets"]["prepared"]))
        command.add_argument("--profile", type=Path, default=Path(__file__).parent / "resources/stage4.yaml")
        command.add_argument("--output", type=Path, required=True)
        if name in ("generate", "compare"):
            command.add_argument("--limit", type=int)
            command.add_argument("--retry-failed", action="store_true")
        if name == "compare":
            command.add_argument("--selection", type=Path, default=Path(__file__).parent / "resources/stage4_pilot.json")
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
    args = parser.parse_args(argv)
    if args.command in ("audit", "generate", "compare", "build"):
        config = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or set(config) != {"generator", "embedding"}:
            raise ValueError("Stage4 profile needs generator and embedding sections")
        profile, embedding = generation_profile(config["generator"]), embedding_profile(config["embedding"])
        train, experiment = data(args.prepared)
        if args.command == "audit":
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "experiment_manifest.json", experiment)
            write_json(args.output / "generation_manifest.json", run_manifest(train, experiment, profile))
            result = {"count": len(train), "sources": train.dataset_source.value_counts().to_dict(), "embedding": embedding}
            write_json(args.output / "audit.json", result)
        elif args.command in ("generate", "compare"):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Generation requires a CUDA accelerator; use the supplied Kaggle notebook")
            write_json(args.output / "experiment_manifest.json", experiment)
            write_json(args.output / "environment.json", {"python": sys.version,
                "packages": {name: importlib.metadata.version(name) for name in ("torch","transformers","sentence-transformers","huggingface-hub","numpy","pandas")},
                "cuda": torch.version.cuda, "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
            generator = HFInsightGenerator(profile)
            if args.command == "compare":
                result = run_comparison(train, experiment, args.output, profile, read_json(args.selection),
                    lambda records, variant: generator.generate_batch(records, prompt_variant=variant),
                    limit=args.limit, retry_failed=args.retry_failed)
            else:
                result = run_insights(train, experiment, args.output, profile, generator.generate_batch,
                                      limit=args.limit, retry_failed=args.retry_failed)
        else:
            result = build_corpus(train, experiment, args.insights, args.output, embedding, Encoder(embedding, device=args.device))
    else:
        corpus = Corpus(args.corpus)
        if args.command == "inspect":
            result = corpus.manifest
        else:
            from ielts_judge.database import ReferenceRetriever, import_corpus
            rpc = rpc_from_env(args)
            if args.command == "ingest":
                result = import_corpus(corpus, rpc)
            elif args.command == "verify-rpc":
                result = verify_rpc(corpus, rpc)
                write_json(args.output, result)
            else:
                retriever = ReferenceRetriever(rpc, corpus.manifest, Encoder(corpus.manifest["embedding"]))
                found = retriever.retrieve(args.essay_file.read_text(encoding="utf-8"))
                result = {**found, "references": [r.__dict__ for r in found["references"]]}
                write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
