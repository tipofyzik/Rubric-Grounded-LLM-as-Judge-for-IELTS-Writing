"""Full-train, resumable PANDA generation with one atomic result per essay ID."""

import importlib.metadata
import os
from pathlib import Path
import time

import numpy as np

from .artifacts import canonical, insights, parse_completion, parse_json, parse_legacy_completion, positive_int, read_json, revision, sha256, write_json
from ..criteria import IELTS_CRITERIA, PUBLIC_CRITERION_IDS
from ..data.experiment import validate_reference_corpus

PROMPT_VERSIONS = {"simple_json": "panda-insights-json-v2", "legacy_text": "panda-insights-text-v1",
                   "original_text": "panda-insights-original-085e4a1-v1",
                   "adjacent_text": "panda-insights-adjacent-text-v5"}
GENERATOR_DEFAULTS = {
    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "max_input_tokens": 4096, "max_new_tokens": 1024, "seed": 42, "prompt_variant": "simple_json",
    "batch_size": 12,
}


def generation_profile(config=None):
    if config is not None and (not isinstance(config, dict) or config.keys() - GENERATOR_DEFAULTS.keys()):
        raise ValueError("Unknown insight generation options")
    result = {**GENERATOR_DEFAULTS, **(config or {})}
    revision(result["revision"])
    if not isinstance(result["model_id"], str) or not result["model_id"].strip():
        raise ValueError("Generator model_id is required")
    for name in ("max_input_tokens", "max_new_tokens", "batch_size"):
        positive_int(result[name], name)
    if type(result["seed"]) is not int or result["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if result["prompt_variant"] not in PROMPT_VERSIONS:
        raise ValueError("Unknown prompt_variant")
    return {**result, "prompt_version": PROMPT_VERSIONS[result["prompt_variant"]],
            "validation_version": "panda-insights-validation-v2",
            "rubric_sha256": sha256(canonical(IELTS_CRITERIA)),
            "rubric": parse_json(canonical(IELTS_CRITERIA)),
            "rubric_source": "project abbreviated IELTS rubric; not verbatim official descriptors",
            "do_sample": False, "dtype": "float16", "input_policy": "full task+essay; reject over budget"}


def result_path(output, essay_id):
    return Path(output) / "responses" / (sha256(essay_id) + ".json")


def row_input(row):
    return {"essay_id": row["essay_id"], "essay_hash": row["essay_hash"],
            "task_prompt": row["task_prompt"], "essay": row["essay"],
            "scores": {name: int(row[name]) for name in PUBLIC_CRITERION_IDS}}


def run_manifest(train, experiment, profile):
    validate_reference_corpus(train, experiment)
    return {"schema_version": "panda-generation-v1", "dataset_manifest_sha256": experiment["dataset_manifest_sha256"],
            "dataset_source": experiment["dataset_source"], "train": experiment["train"], "generator": profile}


def parse_insights(completion, record, profile, *, allow_partial=False):
    """Validate insights by default; allow_partial returns unvalidated data for failed records only."""
    legacy = profile.get("prompt_variant") in ("legacy_text", "original_text", "adjacent_text")
    parsed = parse_legacy_completion(completion, record["scores"],
        explicit_bands=profile.get("prompt_variant") == "adjacent_text", allow_partial=allow_partial) if legacy else parse_completion(completion)
    if allow_partial:
        return (parsed or None) if legacy else parsed
    return insights(parsed, record["essay"], record["scores"], require_quote=not legacy)


def load_result(output, row, manifest):
    path = result_path(output, row["essay_id"])
    if not path.exists():
        return None
    result = read_json(path)
    if result.get("input_sha256") != sha256(canonical(row_input(row))) or result.get("run_sha256") != sha256(canonical(manifest)):
        raise ValueError(f"Stale insight result: {row['essay_id']}")
    if result.get("status") not in ("accepted", "failed"):
        raise ValueError("Unknown insight result status")
    if result["status"] == "accepted":
        if parse_insights(result["completion"], row_input(row), manifest["generator"]) != result["insights"] or result["diagnostics"].get("stop_reason") != "eos":
            raise ValueError("Accepted insight differs from the saved completed response")
    return result


def pilot_order(train):
    """Cover source/score profiles first, then fill by stable ID hash; always retain full membership."""
    groups = {}
    for row in train.to_dict("records"):
        key = (row["dataset_source"], *(int(row[name]) for name in PUBLIC_CRITERION_IDS))
        groups.setdefault(key, []).append(row)
    ordered = []
    for key in sorted(groups):
        groups[key].sort(key=lambda row: (-len(row["essay"]), sha256(row["essay_id"])))
        ordered.append(groups[key].pop(0))
    ordered.extend(sorted((row for values in groups.values() for row in values), key=lambda row: sha256(row["essay_id"])))
    return ordered


def run_insights(train, experiment, output, profile, generate_batch, *, limit=None, retry_failed=False, essay_ids=None):
    """Single writer. A batch returns one completion or ValueError per input, in order."""
    if limit is not None:
        positive_int(limit, "limit")
    output = Path(output)
    if (output / "comparison_manifest.json").exists():
        raise ValueError("Comparison directory cannot be used for full generation")
    manifest = run_manifest(train, experiment, profile)
    rows = pilot_order(train)
    if essay_ids is not None:
        by_id = {row["essay_id"]: row for row in rows}
        if not essay_ids or len(set(essay_ids)) != len(essay_ids) or not set(essay_ids) <= by_id.keys():
            raise ValueError("Selection must contain unique train essay IDs")
        rows = [by_id[essay_id] for essay_id in essay_ids]
        manifest["selection"] = list(essay_ids)
    passport = output / "generation_manifest.json"
    if passport.exists() and read_json(passport) != manifest:
        raise ValueError("Generation profile/train changed; use a new output directory")
    if not passport.exists() and (output / "responses").exists():
        raise ValueError("Orphan responses without generation manifest")
    write_json(passport, manifest)
    batch_size = positive_int(profile.get("batch_size", 1), "batch_size")
    pending = []
    for row in rows:
        previous = load_result(output, row, manifest)
        if previous is not None and (previous["status"] == "accepted" or not retry_failed):
            continue
        pending.append(row)
        if limit is not None and len(pending) >= limit:
            break
    processing_started = time.perf_counter()
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        started = time.perf_counter()
        generated = generate_batch([row_input(row) for row in batch])
        if len(generated) != len(batch):
            raise ValueError("Generator returned a different number of batch results")
        for row, response in zip(batch, generated):
            record = {"essay_id": row["essay_id"], "run_sha256": sha256(canonical(manifest)),
                      "input_sha256": sha256(canonical(row_input(row))), "insights": None, "error": None,
                      "batch_size": len(batch)}
            try:
                if isinstance(response, ValueError):
                    raise response
                completion, diagnostics = response
                record.update(completion=completion, diagnostics=diagnostics)
                if diagnostics.get("stop_reason") != "eos":
                    raise ValueError("Generation did not finish with EOS")
                record.update(insights=parse_insights(completion, row_input(row), profile), status="accepted")
            except ValueError as error:
                record.update(status="failed", error=str(error))
                if "completion" in record:
                    try:
                        record["insights"] = parse_insights(record["completion"], row_input(row), profile, allow_partial=True)
                    except ValueError:
                        # Unparseable JSON stays null; completion and the original error are preserved.
                        pass
            record["seconds"] = time.perf_counter() - started
            write_json(result_path(output, row["essay_id"]), record)
    processing_seconds = time.perf_counter() - processing_started
    counts = {"accepted": 0, "failed": 0, "pending": 0}
    for row in rows:
        result = load_result(output, row, manifest)
        counts[result["status"] if result else "pending"] += 1
    report = {"schema_version": "panda-generation-report-v1", "count": len(rows), **counts,
              "attempted_this_invocation": len(pending), "complete": counts["accepted"] == len(rows),
              "processing_seconds_this_invocation": processing_seconds,
              "seconds_per_attempt_this_invocation": processing_seconds / len(pending) if pending else None}
    if essay_ids is not None:
        report.update(scope="comparison_selection_only", full_train_count=len(train))
    write_json(output / "generation_report.json", report)
    return report


class HFInsightGenerator:
    """Loads pinned weights only on the first missing essay; generate never truncates the prompt."""

    def __init__(self, profile):
        self.profile = profile
        self.runtime = None

    def __call__(self, record, *, prompt_variant=None):
        result = self.generate_batch([record], prompt_variant=prompt_variant)[0]
        if isinstance(result, ValueError):
            raise result
        return result

    def generate_batch(self, records, *, prompt_variant=None):
        """Left-pad valid inputs together; retain individual budget failures and EOS lengths."""
        import torch
        from transformers import set_seed
        from .InsightGenerator import PandaInsightGenerator

        if not records:
            return []
        if self.runtime is None:
            if not torch.cuda.is_available():
                raise RuntimeError("PANDA generation requires a CUDA GPU; select a Kaggle accelerator")
            self.runtime = PandaInsightGenerator(os.environ.get("HF_TOKEN", ""))
            self.runtime.load_model_and_tokenizer(self.profile["model_id"], revision=self.profile["revision"])
            self.runtime.model.eval()
        runtime = self.runtime
        variant = prompt_variant or self.profile["prompt_variant"]
        if variant not in PROMPT_VERSIONS:
            raise ValueError("Unknown prompt_variant")
        results: dict[int, tuple[str, dict] | ValueError] = {}
        valid = []
        features = []
        capacity = getattr(runtime.model.config, "max_position_embeddings", None)
        for index, record in enumerate(records):
            try:
                prompt = runtime.build_insight_prompt(record["essay"], np.array(list(record["scores"].values())),
                                                      record["task_prompt"], structured=variant == "simple_json",
                                                      simple=variant == "simple_json", original=variant == "original_text",
                                                      explicit_bands=variant == "adjacent_text")
                text = runtime.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                             tokenize=False, add_generation_prompt=True)
                encoded = runtime.tokenizer(text, return_tensors="pt", truncation=False)
                length = encoded["input_ids"].shape[1]
                if length > self.profile["max_input_tokens"] or (capacity and length + self.profile["max_new_tokens"] > capacity):
                    raise ValueError(f"Input token budget exceeded: {length}; no task/essay truncation was applied")
                valid.append((index, length))
                features.append({name: value[0] for name, value in encoded.items()})
            except ValueError as error:
                results[index] = error
        if not valid:
            return [results[index] for index in range(len(records))]
        inputs = runtime.tokenizer.pad(features, padding=True, return_tensors="pt").to(runtime.model.device)
        width = inputs["input_ids"].shape[1]
        seed = (self.profile["seed"] + int(sha256(records[valid[0][0]]["essay_id"])[:8], 16)) % (2**32)
        set_seed(seed)
        with torch.inference_mode():
            outputs = runtime.model.generate(**inputs, max_new_tokens=self.profile["max_new_tokens"],
                                             do_sample=False, pad_token_id=runtime.tokenizer.pad_token_id)
        if len(outputs) != len(valid):
            raise ValueError("Model returned a different number of batch sequences")
        outputs = outputs[:, width:].cpu().tolist()
        eos = runtime.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "huggingface-hub")}
        for (index, length), tokens in zip(valid, outputs):
            end = next((i + 1 for i, token in enumerate(tokens) if token in eos), None)
            if end is not None:
                tokens = tokens[:end]
            completion = runtime.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            diagnostics = {"input_tokens": length, "output_tokens": len(tokens),
                           "stop_reason": "eos" if end is not None else "length",
                           "seed": seed, "device": str(runtime.model.device), "versions": versions,
                           "batch_size": len(valid), "padded_input_tokens": width}
            results[index] = completion, diagnostics
        return [results[index] for index in range(len(records))]
