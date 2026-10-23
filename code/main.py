"""Run explicitly enabled research stages against a verified local Parquet release."""

from pathlib import Path
import json
from uuid import uuid4

import yaml

from ielts_judge.paths import PATHS
from ielts_judge.criteria import CRITERION_IDS
from ielts_judge.data.DatasetTransformer import DatasetTransformer
from ielts_judge.data.preparation import legacy_labels
from ielts_judge.data.experiment import prepare_experiment
from ielts_judge.data.runtime import (
    convert_to_llm_dataset,
    export_reference,
    submission_records,
)

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_data(config):
    dataset_config = config.get("dataset", {})
    transformer = DatasetTransformer(
        PATHS["datasets"],
        selection=dataset_config.get("selection"),
        save_intermediate_datasets=dataset_config.get(
            "save_intermediate_datasets", False
        ),
    )
    if config["stage_switch"].get("transform_datasets"):
        transformer.prepare_and_save(PATHS["datasets"]["prepared"])
    return transformer.load_processed(
        PATHS["datasets"]["prepared"],
        config.get("dataset", {}).get("source", "combined"),
    )


def main(*, prepare_only=False, selection_manifest=None, run_directory=None):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    switch = config["stage_switch"]
    train, test_pool, manifest = load_data(config)
    print(
        f"Verified Parquet: train={len(train):,}, test pool={len(test_pool):,}; source={config.get('dataset', {}).get('source', 'combined')}"
    )
    release = Path(PATHS["datasets"]["prepared"])
    if switch.get("analyze_data"):
        print((release / "audit.json").read_text(encoding="utf-8"))
        profiles = json.loads(
            (release / "score_profile_counts.json").read_text(encoding="utf-8")
        )
        # Full ID memberships stay in the report, not in the console output.
        print(
            json.dumps(
                {
                    name: profiles[name]
                    for name in (
                        "schema_version",
                        "analyzed_essays",
                        "counts",
                        "direction_counts",
                        "type_totals",
                        "not_matching_any_type",
                    )
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if selection_manifest is not None and run_directory is not None:
        raise ValueError("Use either --selection-manifest or --run-directory")
    saved_path = (Path(run_directory) / "experiment_manifest.json" if run_directory is not None
                  else Path(selection_manifest) if selection_manifest is not None else None)
    saved = json.loads(saved_path.read_text(encoding="utf-8")) if saved_path is not None else None
    test, experiment = prepare_experiment(
        train, test_pool, config=config.get("evaluation"),
        source=config.get("dataset", {}).get("source", "combined"), saved_manifest=saved,
    )
    if run_directory is None:
        result_root = (Path(PATHS["results"]["dataset_runs"]) /
                       train.attrs["manifest_sha256"][:12] / uuid4().hex)
        result_root.mkdir(parents=True, exist_ok=False)
        (result_root / "experiment_manifest.json").write_text(
            json.dumps(experiment, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    else:
        result_root = Path(run_directory)
    print(f"Experiment: full train/reference={len(train):,}; selected test={len(test):,}; {result_root}")
    active = not prepare_only and any(
        switch.get(name)
        for name in (
            "train_custom_regressor",
            "fine_tune_llm",
            "get_llm_predictions",
            "panda_evaluation",
            "aggregate_results",
            "evaluate_llm",
        )
    )
    if not active:
        return train, test, manifest
    if train.empty or test.empty:
        raise ValueError("Model stages require nonempty train and test splits")
    import numpy as np

    # Separate release outputs from historic CSV experiments.
    PATHS["results"].update(
        llm_hugging_face_responses=str(result_root / "responses"),
        llm_training=str(result_root / "training"),
        llm_predictions=str(result_root / "fine_tuned_predictions.json"),
        panda_insights=str(result_root / "panda_insights.csv"),
    )
    reference_path = export_reference(test, result_root / "y_test.csv")
    model_config = config.get("model", {})
    model_id, token = model_config.get("id", ""), model_config.get("token", "")
    PATHS["files"]["llm_response"] = (
        model_config.get("key", "qwen25") + "_{evaluation}.json"
    )
    train_records, test_records = submission_records(train), submission_records(test)
    y_train, y_test = legacy_labels(train), legacy_labels(test)
    if switch.get("train_custom_regressor"):
        from ielts_judge.aes.OrdinalRegression import OrdinalRegression

        model = OrdinalRegression(num_classes=10, num_tasks=len(CRITERION_IDS))
        band2idx = {i: i for i in range(10)}
        encoded = model.ordinal_encode_multi(y_train, band2idx, 10)
        model.fit(train.essay.tolist(), encoded)
        predictions = model.predict(test.essay.tolist(), band2idx)
        import pandas as pd

        table = pd.DataFrame(
            predictions, columns=[f"pred_{name}" for name in CRITERION_IDS]
        )
        table.insert(0, "essay_id", test.essay_id.to_numpy())
        table["dataset_manifest"] = train.attrs["manifest_sha256"]
        table.to_csv(result_root / "aes_predictions.csv", index=False)
    if switch.get("fine_tune_llm"):
        from ielts_judge.llm.LLMFineTuner import LLMFineTuner

        trainer = LLMFineTuner(model_name=model_id, max_len=1536, mode="lora")
        trainer.train(train_records, y_train.to_numpy().tolist())
        trainer.evaluate_llm(
            test_records,
            y_test.to_numpy().tolist(),
            output_file=str(result_root / "fine_tuned_predictions.json"),
        )
    if switch.get("get_llm_predictions"):
        from ielts_judge.llm.LLMResponseObtainer import LLMResponseObtainer

        for name, frame, is_test in (("train", train, False), ("test", test, True)):
            convert_to_llm_dataset(frame, test_data=is_test).to_parquet(
                result_root / f"{name}_llm.parquet", index=False
            )
        obtainer = LLMResponseObtainer(SERVICE_TOKEN=token)
        obtainer.load_model_and_tokenizer(model_url=model_id)
        obtainer.get_response_locally(
            number_of_evaluations=config.get("inference", {}).get("runs", 1),
            test_dataset=test,
        )
    if switch.get("panda_evaluation"):
        from ielts_judge.panda.Panda import PANDA

        panda = PANDA(service_token=token)
        panda.load_model_and_tokenizer(model_url=model_id)
        panda.retrieve_insights(
            essays=np.array(train.essay),
            scores=y_train.to_numpy(),
            submissions=train_records,
        )
    if switch.get("aggregate_results"):
        from ielts_judge.evaluation.LLMAggregator import LLMAggregator

        LLMAggregator(evaluation_folder_path=str(result_root / "responses")).run(
            str(result_root / "predictions.csv")
        )
    if switch.get("evaluate_llm"):
        from ielts_judge.evaluation.LLMEvaluator import LLMEvaluator

        evaluator = LLMEvaluator(
            results_folder_path=str(result_root), y_true_path=str(reference_path)
        )
        evaluator.load_all(run_order=["predictions"])
        evaluator.plot_metrics(save_dir=str(result_root / "plots"))
    return train, test, manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Save selection metadata without model stages")
    parser.add_argument("--selection-manifest", type=Path, help="Replay a saved selection in a new run directory")
    parser.add_argument("--run-directory", type=Path, help="Use an existing run and its saved selection")
    args = parser.parse_args()
    main(prepare_only=args.prepare_only, selection_manifest=args.selection_manifest, run_directory=args.run_directory)
