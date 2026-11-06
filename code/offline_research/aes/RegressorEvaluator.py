"""AES entry points using the same integer metrics and reports as judge models."""
from typing import Any

from pathlib import Path

import pandas as pd

from ielts_judge.Criteria import LEGACY_TO_PUBLIC, PUBLIC_CRITERION_IDS
from offline_research.evaluation.Metrics import integer_bands, metric_values


class RegressorEvaluator:
    """Manage regressor evaluator through the operations defined by this component."""
    def get_metrics_per_criterion(self, y_true: Any, y_pred: Any, criteria: Any) -> Any:
        """Get metrics per criterion.

        Args:
            y_true (Any): Y true used by this operation.
            y_pred (Any): Y pred used by this operation.
            criteria (Any): Criteria used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        actual, predicted = integer_bands(y_true), integer_bands(y_pred)
        if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[1] != len(criteria):
            raise ValueError("AES predictions and reference shapes differ")
        result = {}
        for current_index, criterion in enumerate(criteria):
            values, reasons = metric_values(actual[:, current_index], predicted[:, current_index])
            result[criterion] = {**values, "undefined_reasons": reasons, "count": len(actual)}
        return result

    def plot_loss_curve(self, loss_history: Any, save_path: Any) -> None:
        """Plot loss curve.

        Args:
            loss_history (Any): Loss history used by this operation.
            save_path (Any): Save path used by this operation.

        Returns:
            None: This operation returns no value.
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        try:
            ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
            ax.set(xlabel="Epoch", ylabel="Loss", title="AES training loss")
            fig.tight_layout()
            fig.savefig(save_path, dpi=150)
        finally:
            plt.close(fig)

    def save_predictions_table(self, y_true: Any, y_pred: Any, criteria: Any, save_path: Any, *, essay_ids: Any=None) -> None:
        """Save predictions table.

        Args:
            y_true (Any): Y true used by this operation.
            y_pred (Any): Y pred used by this operation.
            criteria (Any): Criteria used by this operation.
            save_path (Any): Save path used by this operation.
            essay_ids (Any): Essay ids used by this operation.

        Returns:
            None: This operation returns no value.
        """
        actual, predicted = integer_bands(y_true), integer_bands(y_pred)
        if actual.shape != predicted.shape:
            raise ValueError("AES predictions and reference shapes differ")
        table = pd.DataFrame({"essay_id": list(essay_ids) if essay_ids is not None else list(map(str, range(len(actual))))})
        for current_index, criterion in enumerate(criteria):
            public = LEGACY_TO_PUBLIC.get(criterion, criterion)
            table["true_" + public] = actual[:, current_index]
            table["pred_" + public] = predicted[:, current_index]
        path = Path(save_path)
        if path.exists():
            raise FileExistsError(path)
        table.to_csv(path, index=False)

    def run_full_evaluation(self, y_true: Any, y_pred: Any, criteria: Any, bands: Any, save_dir: Any, loss_history: Any=None,
                            *, essay_ids: Any=None, report_config: Any=None) -> Any:
        """Run full evaluation.

        Args:
            y_true (Any): Y true used by this operation.
            y_pred (Any): Y pred used by this operation.
            criteria (Any): Criteria used by this operation.
            bands (Any): Bands used by this operation.
            save_dir (Any): Save dir used by this operation.
            loss_history (Any): Loss history used by this operation.
            essay_ids (Any): Essay ids used by this operation.
            report_config (Any): Report config used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from offline_research.evaluation.Report import write_report
        actual, predicted = integer_bands(y_true), integer_bands(y_pred)
        if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[1] != 4:
            raise ValueError("AES report requires aligned four-criterion predictions")
        if set(LEGACY_TO_PUBLIC.get(criterion, criterion) for criterion in criteria) != set(PUBLIC_CRITERION_IDS):
            raise ValueError("AES report requires the four public criteria")
        ids = list(map(str, essay_ids)) if essay_ids is not None else list(map(str, range(len(actual))))
        truth = pd.DataFrame(actual, columns=[LEGACY_TO_PUBLIC.get(criterion, criterion) for criterion in criteria])
        truth.insert(0, "essay_id", ids)
        records = [{"model_key": "aes", "setup": "aes", "repeat": 0, "essay_id": essay_id, "status": "ok",
                    "scores": dict(zip([LEGACY_TO_PUBLIC.get(criterion, criterion) for criterion in criteria], predicted[current_index].tolist()))}
                   for current_index, essay_id in enumerate(ids)]
        plan = {"models": [], "setups": [], "repeats": 1, "essay_ids": ids, "include_aes": True}
        result = write_report(records, truth, plan, save_dir, report_config)
        if loss_history:
            self.plot_loss_curve(loss_history, result / "aes_training_loss.png")
        return result
