"""AES entry points using the same integer metrics and reports as judge models."""

from pathlib import Path

import pandas as pd

from ..criteria import LEGACY_TO_PUBLIC, PUBLIC_CRITERION_IDS
from ..evaluation.metrics import integer_bands, metric_values


class RegressorEvaluator:
    def get_metrics_per_criterion(self, y_true, y_pred, criteria):
        actual, predicted = integer_bands(y_true), integer_bands(y_pred)
        if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[1] != len(criteria):
            raise ValueError("AES predictions and reference shapes differ")
        result = {}
        for i, criterion in enumerate(criteria):
            values, reasons = metric_values(actual[:, i], predicted[:, i])
            result[criterion] = {**values, "undefined_reasons": reasons, "count": len(actual)}
        return result

    def plot_loss_curve(self, loss_history, save_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        try:
            ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
            ax.set(xlabel="Epoch", ylabel="Loss", title="AES training loss")
            fig.tight_layout()
            fig.savefig(save_path, dpi=150)
        finally:
            plt.close(fig)

    def save_predictions_table(self, y_true, y_pred, criteria, save_path, *, essay_ids=None):
        actual, predicted = integer_bands(y_true), integer_bands(y_pred)
        if actual.shape != predicted.shape:
            raise ValueError("AES predictions and reference shapes differ")
        table = pd.DataFrame({"essay_id": list(essay_ids) if essay_ids is not None else list(map(str, range(len(actual))))})
        for i, criterion in enumerate(criteria):
            public = LEGACY_TO_PUBLIC.get(criterion, criterion)
            table["true_" + public] = actual[:, i]
            table["pred_" + public] = predicted[:, i]
        path = Path(save_path)
        if path.exists():
            raise FileExistsError(path)
        table.to_csv(path, index=False)

    def run_full_evaluation(self, y_true, y_pred, criteria, bands, save_dir, loss_history=None,
                            *, essay_ids=None, report_config=None):
        from ..evaluation.report import write_report
        actual, predicted = integer_bands(y_true), integer_bands(y_pred)
        if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[1] != 4:
            raise ValueError("AES report requires aligned four-criterion predictions")
        if set(LEGACY_TO_PUBLIC.get(c, c) for c in criteria) != set(PUBLIC_CRITERION_IDS):
            raise ValueError("AES report requires the four public criteria")
        ids = list(map(str, essay_ids)) if essay_ids is not None else list(map(str, range(len(actual))))
        truth = pd.DataFrame(actual, columns=[LEGACY_TO_PUBLIC.get(c, c) for c in criteria])
        truth.insert(0, "essay_id", ids)
        records = [{"model_key": "aes", "setup": "aes", "repeat": 0, "essay_id": essay_id, "status": "ok",
                    "scores": dict(zip([LEGACY_TO_PUBLIC.get(c, c) for c in criteria], predicted[i].tolist()))}
                   for i, essay_id in enumerate(ids)]
        plan = {"models": [], "setups": [], "repeats": 1, "essay_ids": ids, "include_aes": True}
        result = write_report(records, truth, plan, save_dir, report_config)
        if loss_history:
            self.plot_loss_curve(loss_history, result / "aes_training_loss.png")
        return result
