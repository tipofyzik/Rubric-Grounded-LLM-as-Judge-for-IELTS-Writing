"""CSV entry point for the shared model evaluation tables and plots."""

from pathlib import Path

import pandas as pd

from ..criteria import CRITERION_IDS, CRITERION_NAMES, LEGACY_TO_PUBLIC, PUBLIC_CRITERION_IDS
from .config import report_options
from .metrics import metric_values
from .report import write_report


class LLMEvaluator:
    def __init__(self, results_folder_path, y_true_path, dpi=None, aes_expert_metrics=None,
                 *, aes_predictions=None, report_config=None):
        self.results_folder_path = results_folder_path
        self.y_true_df = pd.read_csv(y_true_path).sort_values("index").reset_index(drop=True)
        self.aes_expert_metrics = aes_expert_metrics
        self.aes_predictions = aes_predictions
        self.report_config = report_options(report_config)
        self.dpi = self.report_config["dpi"] if dpi is None else dpi
        self.all_data, self.run_order = {}, None
        self.criteria = list(CRITERION_IDS)
        self.true_criteria_names = CRITERION_NAMES
        self.true_names = {
            "non_fine_tuned_raw_result": "base", "non_fine_tuned_raw_rubric_result": "rubric",
            "non_fine_tuned_raw_panda_result": "panda", "non_fine_tuned_raw_rubric_panda_result": "rubric_panda",
            "non_fine_tuned_raw_panda_scores_result": "panda_aes",
            "non_fine_tuned_raw_rubric_panda_scores_result": "rubric_panda_aes",
        }
        self.true_model_names = {"qwen25": "Qwen2.5 1.5B Instruct", "qwen25_1p5b": "Qwen2.5 1.5B Instruct",
                                 "qwen2": "Qwen2 1.5B Instruct", "llama_32_3B": "Llama 3.2 3B Instruct",
                                 "mistral_7B": "Mistral 7B Instruct v0.2"}

    def load_all(self, run_order=None):
        self.all_data, self.run_order = {}, run_order
        folder = Path(self.results_folder_path)
        files = [folder / (name + ".csv") for name in run_order] if run_order is not None else sorted(folder.glob("*.csv"))
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(f"Requested prediction CSV is missing: {path}")
            frame = pd.read_csv(path)
            if not {"model", "file_id", "sample_id"}.issubset(frame):
                if run_order is not None:
                    raise ValueError(f"Prediction CSV has no model/file_id/sample_id: {path}")
                continue
            self.all_data[path.stem] = {model: {int(run): group.sort_values("sample_id").reset_index(drop=True)
                                              for run, group in model_frame.groupby("file_id")}
                                       for model, model_frame in frame.groupby("model")}

    def _get_all_models(self):
        return sorted({model for data in self.all_data.values() for model in data})

    def _get_true_pred(self, run_df, crit):
        if "dataset_manifest" in self.y_true_df:
            expected = set(self.y_true_df.dataset_manifest.dropna())
            if (len(expected) != 1 or self.y_true_df.dataset_manifest.isna().any()
                    or "dataset_manifest" not in run_df or run_df.dataset_manifest.isna().any()
                    or set(run_df.dataset_manifest) != expected):
                raise ValueError("Predictions and reference belong to different dataset releases")
        if "sample_id" not in run_df or "index" not in self.y_true_df:
            raise ValueError("Both prediction sample_id and reference index are required")
        if run_df.sample_id.duplicated().any() or self.y_true_df["index"].duplicated().any():
            raise ValueError("Prediction and reference IDs must be unique")
        aligned = run_df[["sample_id", crit]].merge(self.y_true_df[["index", crit]], left_on="sample_id", right_on="index",
            how="outer", validate="one_to_one", indicator=True, suffixes=("_pred", "_true"))
        if not aligned["_merge"].eq("both").all():
            raise ValueError(f"Prediction sample IDs do not match reference IDs for {crit}")
        return aligned[crit + "_true"].to_numpy(dtype=float), aligned[crit + "_pred"].to_numpy(dtype=float)

    def _qwk(self, actual, predicted):
        return metric_values(actual, predicted)[0]["qwk"]

    def _mae(self, actual, predicted):
        return metric_values(actual, predicted)[0]["mae"]

    def _rmse(self, actual, predicted):
        return metric_values(actual, predicted)[0]["rmse"]

    def _pearson_corr(self, actual, predicted):
        return metric_values(actual, predicted)[0]["pearson"]

    def _spearman_corr(self, actual, predicted):
        return metric_values(actual, predicted)[0]["spearman"]

    def plot_metrics(self, save_dir):
        if not self.all_data:
            raise ValueError("No prediction runs loaded")
        if self.aes_expert_metrics is not None and self.aes_predictions is None:
            raise ValueError("AES comparison requires ID-aligned predictions; aggregate-only metrics cannot verify the sample")
        models = self._get_all_models()
        setups = list(self.all_data)
        repeats = max(len(runs) for data in self.all_data.values() for runs in data.values())
        truth = self.y_true_df.rename(columns={"index": "essay_id", **LEGACY_TO_PUBLIC})
        manifest = truth.dataset_manifest.iloc[0] if "dataset_manifest" in truth else None
        records = []
        for setup, data in self.all_data.items():
            for model, runs in data.items():
                for repeat, (_, frame) in enumerate(sorted(runs.items())):
                    for criterion in self.criteria:
                        self._get_true_pred(frame, criterion)
                    for row in frame.to_dict("records"):
                        scores = {LEGACY_TO_PUBLIC[c]: row[c] for c in self.criteria}
                        records.append({"essay_id": str(row["sample_id"]), "model_key": model, "setup": setup, "repeat": repeat,
                                        "dataset_manifest": row.get("dataset_manifest"), "status": "ok", "scores": scores})
        if self.aes_predictions is not None:
            aes = self.aes_predictions.rename(columns={"index": "essay_id", **LEGACY_TO_PUBLIC})
            for row in aes.to_dict("records"):
                scores = {c: row.get("pred_" + c, row.get(c)) for c in PUBLIC_CRITERION_IDS}
                records.append({"essay_id": str(row["essay_id"]), "model_key": "aes", "setup": "aes", "repeat": 0,
                                "dataset_manifest": row.get("dataset_manifest"), "status": "ok", "scores": scores})
        plan = {"models": models, "model_labels": {m: self.true_model_names.get(m, m) for m in models},
                "setups": setups, "repeats": repeats, "essay_ids": truth.essay_id.astype(str).tolist(),
                "dataset_manifest_sha256": manifest, "include_aes": self.aes_predictions is not None}
        options = {**self.report_config, "dpi": self.dpi}
        return write_report(records, truth, plan, save_dir, options)
