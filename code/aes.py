"""Train, audit or predict with an essay-only AES bundle (no judge/retrieval)."""

import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "audit"):
        command = subcommands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
        command.add_argument("--output", type=Path)
        if name == "train":
            command.add_argument("--require-cuda", action="store_true")
    predict = subcommands.add_parser("predict")
    predict.add_argument("--bundle", type=Path, required=True)
    predict.add_argument("--essay-file", type=Path, required=True)
    predict.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    from ielts_judge.aes.OrdinalRegression import OrdinalRegression, length_summary, token_lengths
    from ielts_judge.aes.inference import write_json
    if args.command == "predict":
        model = OrdinalRegression.load_bundle(args.bundle, device=args.device)
        result = model.predict_essay(args.essay_file.read_text(encoding="utf-8"))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return result
    import torch
    from transformers import AutoTokenizer
    from ielts_judge.aes.training import aes_options, run_aes
    from ielts_judge.data.experiment import prepare_experiment
    from main import load_data
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["stage_switch"].get("transform_datasets"):
        raise ValueError("AES commands require an existing release; set transform_datasets: 0")
    if args.command == "train" and args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: select a GPU accelerator in Kaggle")
    train, pool, _ = load_data(config)
    selected, experiment = prepare_experiment(train, pool, config=config.get("evaluation"),
                                             source=config.get("dataset", {}).get("source", "combined"))
    output = args.output or Path("results/stage3") / uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment_manifest.json", experiment)
    options = aes_options(config.get("aes"))
    if args.command == "audit":
        tokenizer = AutoTokenizer.from_pretrained(options["encoder_name"], revision=options["encoder_revision"])
        result = {name: length_summary(token_lengths(tokenizer, frame.essay.tolist()), options["max_length"])
                  for name, frame in (("train", train), ("test_pool", pool), ("selected_test", selected))}
        write_json(output / "length_audit.json", result)
    else:
        result = run_aes(train, selected, experiment, output, config=options)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
