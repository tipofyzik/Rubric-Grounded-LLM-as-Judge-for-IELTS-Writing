import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    confusion_matrix, 
    cohen_kappa_score
    )
from scipy.stats import pearsonr, spearmanr


class RegressorEvaluator:
    """
    Utility class for evaluating ordinal regression model predictions
    and generating evaluation plots.
    """

    def __init__(self) -> None:
        """
        Initializes the RegressorEvaluator instance.
        """

    # ===================================================
    # HELPER: snap continuous predictions to nearest band
    # ===================================================
    def snap_to_bands(self, y: np.ndarray, bands: np.ndarray, 
                      scale_for_int: bool = False) -> np.ndarray:
        """
        Snaps continuous predictions to the nearest discrete band values.

        Arguments:
            y (np.ndarray): Array of predictions or labels with shape
                (num_samples, num_criteria).
            bands (np.ndarray): Array of possible discrete band values.
            scale_for_int (bool): If True, multiplies the snapped values by 2
                and converts them to integers (used for QWK and confusion matrix).

        Returns:
            np.ndarray: Snapped predictions with the same shape as input.
        """
        y = np.asarray(y, dtype=float)
        bands = np.asarray(bands, dtype=float)
        snapped = np.zeros_like(y, dtype=float)
        for i in range(y.shape[1]):
            snapped[:, i] = bands[np.abs(y[:, i][:, None] - bands).argmin(axis=1)]
        if scale_for_int:
            # Умножаем на 2 и округляем до int
            snapped = np.rint(snapped * 2).astype(int)
        return snapped

    # ==========
    # LOSS CURVE
    # ==========
    def plot_loss_curve(self, loss_history: list[float], save_path: str) -> None:
        """
        Plots and saves the training loss curve.

        Arguments:
            loss_history (list[float]): List containing loss values for each epoch.
            save_path (str): File path where the plot will be saved.

        Returns:
            None
        """
        plt.figure(figsize=(6,4))
        plt.plot(range(1,len(loss_history)+1), loss_history, marker='o')
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(range(1,len(loss_history)+1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    # =================
    # MAE per criterion
    # =================
    def _annotate_bars(self, ax, values, fmt="{:.3f}", offset=0.02):
        """
        Adds value labels above bars.
        """
        for i, v in enumerate(values):
            ax.text(
                i,
                v + offset,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=9
            )
            
    def save_mae_barplot(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         criteria: list[str], save_path: str) -> None:
        """
        Computes and saves a bar plot of MAE for each evaluation criterion.

        Arguments:
            y_true (np.ndarray): Ground truth scores with shape
                (num_samples, num_criteria).
            y_pred (np.ndarray): Predicted scores with shape
                (num_samples, num_criteria).
            criteria (list[str]): Names of evaluation criteria.
            save_path (str): File path where the plot will be saved.

        Returns:
            None
        """
        maes = [
            mean_absolute_error(y_true[:, i], y_pred[:, i])
            for i in range(len(criteria))
        ]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(criteria, maes)

        ax.set_title("MAE per Criterion")
        ax.set_ylabel("MAE (Band)")
        ax.tick_params(axis='x', rotation=20)

        self._annotate_bars(ax, maes, fmt="{:.3f}")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    # =================
    # MSE per criterion
    # =================
    def save_mse_barplot(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         criteria: list[str], save_path: str) -> None:
        """
        Computes and saves a bar plot of MSE for each evaluation criterion.

        Arguments:
            y_true (np.ndarray): Ground truth scores with shape
                (num_samples, num_criteria).
            y_pred (np.ndarray): Predicted scores with shape
                (num_samples, num_criteria).
            criteria (list[str]): Names of evaluation criteria.
            save_path (str): File path where the plot will be saved.

        Returns:
            None
        """
        mses = [
            mean_squared_error(y_true[:, i], y_pred[:, i])
            for i in range(len(criteria))
        ]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(criteria, mses, color='orange')

        ax.set_title("MSE per Criterion")
        ax.set_ylabel("MSE (Band²)")
        ax.tick_params(axis='x', rotation=20)

        self._annotate_bars(ax, mses, fmt="{:.3f}")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    # =============================
    # Error histogram per criterion
    # =============================
    def save_error_histograms(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              criteria: list[str], save_dir: str) -> None:
        """
        Generates and saves prediction error histograms for each criterion.

        Arguments:
            y_true (np.ndarray): Ground truth scores with shape
                (num_samples, num_criteria).
            y_pred (np.ndarray): Predicted scores with shape
                (num_samples, num_criteria).
            criteria (list[str]): Names of evaluation criteria.
            save_dir (str): Directory where histogram plots will be saved.

        Returns:
            None
        """
        for i, name in enumerate(criteria):
            errors = y_pred[:,i] - y_true[:,i]
            plt.figure(figsize=(6,4))
            plt.hist(errors, bins=20, alpha=0.7)
            plt.title(f"Error Histogram: {name}")
            plt.xlabel("Prediction Error (Band)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/error_hist_{name}.png", dpi=300)
            plt.close()

    # ======================================
    # Quadratic Weighted Kappa per criterion
    # ======================================
    def save_qwk_barplot(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         criteria: list[str], save_path: str) -> None:
        """
        Computes and saves a bar plot of Quadratic Weighted Kappa (QWK)
        for each evaluation criterion.

        Arguments:
            y_true (np.ndarray): Ground truth integer labels with shape
                (num_samples, num_criteria).
            y_pred (np.ndarray): Predicted integer labels with shape
                (num_samples, num_criteria).
            criteria (list[str]): Names of evaluation criteria.
            save_path (str): File path where the plot will be saved.

        Returns:
            None
        """
        qwks = [
            cohen_kappa_score(y_true[:, i], y_pred[:, i], weights="quadratic")
            for i in range(len(criteria))
        ]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(criteria, qwks, color='green')

        ax.set_title("QWK per Criterion")
        ax.set_ylabel("QWK")
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=20)

        self._annotate_bars(ax, qwks, fmt="{:.3f}", offset=0.01)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    # ==============================
    # Confusion matrix per criterion
    # ==============================
    def save_confusion_matrix_with_original_labels(self, y_true_int: np.ndarray, y_pred_int: np.ndarray, 
                                                   y_true_float: np.ndarray, criteria_name: str, save_path: str) -> None:
        """
        Computes and saves a confusion matrix for a specific evaluation criterion.

        Arguments:
            y_true_int (np.ndarray): Ground truth integer labels.
            y_pred_int (np.ndarray): Predicted integer labels.
            y_true_float (np.ndarray): Original floating-point labels used
                for displaying axis tick labels.
            criteria_name (str): Name of the evaluation criterion.
            save_path (str): File path where the confusion matrix plot will be saved.

        Returns:
            None
        """
        labels = np.sort(np.unique(y_true_int))  # все целые метки
        cm = confusion_matrix(y_true_int, y_pred_int, labels=labels)
        
        plt.figure(figsize=(6,5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, label="Count")
        
        # Подписи на осях — оригинальные оценки
        unique_float_labels = np.sort(np.unique(y_true_float))
        plt.xticks(np.arange(len(labels)), [f"{x:.1f}" for x in unique_float_labels], rotation=45)
        plt.yticks(np.arange(len(labels)), [f"{x:.1f}" for x in unique_float_labels])
        
        # Добавляем числа в ячейки
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, str(cm[i,j]), ha='center', va='center', color='black')
        
        plt.xlabel("Predicted Score")
        plt.ylabel("True Score")
        plt.title(f"Confusion Matrix: {criteria_name}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    # ===============================
    # Calibration curve per criterion
    # ===============================
    def save_calibration_curve_binned(self, y_true, y_pred, criteria, save_dir, n_bins=10):

        plt.figure(figsize=(6, 6))
        bins = np.linspace(0, 9, n_bins + 1)

        for c_idx, crit in enumerate(criteria):

            pred_means = []
            true_means = []

            for j in range(n_bins):
                mask = (
                    (y_pred[:, c_idx] >= bins[j]) &
                    (y_pred[:, c_idx] < bins[j + 1])
                )

                if np.sum(mask) == 0:
                    continue

                pred_means.append(np.mean(y_pred[:, c_idx][mask]))
                true_means.append(np.mean(y_true[:, c_idx][mask]))

            plt.plot(pred_means, true_means, 'o-', label=crit.replace("_", " ").title())

        plt.plot([0, 9], [0, 9], '--', color='black', label="Perfect Calibration")

        plt.xlim(0, 9)
        plt.ylim(0, 9)
        plt.xticks(np.arange(0, 10, 1))
        plt.yticks(np.arange(0, 10, 1))
        plt.grid(True, alpha=0.7)

        plt.legend()
        plt.title("Model calibration curve across all criteria")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/calibration_all.png", dpi=300)
        plt.close()

    def save_predictions_table(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               criteria: list[str], save_path: str) -> None:
        """
        Saves ground truth and predicted values into a table for reproducible evaluation.

        Arguments:
            y_true (np.ndarray): Ground truth scores with shape
                (num_samples, num_criteria).
            y_pred (np.ndarray): Predicted scores with shape
                (num_samples, num_criteria).
            criteria (list[str]): Names of evaluation criteria.
            save_path (str): File path where the CSV table will be saved.

        Returns:
            None
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        data = {}

        # sample id
        data["sample_id"] = np.arange(len(y_true))

        # true scores
        for i, name in enumerate(criteria):
            data[f"true_{name}"] = y_true[:, i]

        # predicted scores
        for i, name in enumerate(criteria):
            data[f"pred_{name}"] = y_pred[:, i]

        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)

    
    def save_rmse_barplot(self, y_true: np.ndarray, y_pred: np.ndarray,
                      criteria: list[str], save_path: str) -> None:
        rmses = [
            np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            for i in range(len(criteria))
        ]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(criteria, rmses, color='red')

        ax.set_title("RMSE per Criterion")
        ax.set_ylabel("RMSE (Band)")
        ax.tick_params(axis='x', rotation=20)

        self._annotate_bars(ax, rmses, fmt="{:.3f}")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def save_adjacent_accuracy_barplot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    criteria: list[str], save_path: str) -> None:
        adj_acc = []
        for i in range(len(criteria)):
            diff = np.abs(y_true[:, i] - y_pred[:, i])
            adj_acc.append(np.mean(diff <= 1))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(criteria, adj_acc, color='purple')

        ax.set_title("Adjacent Accuracy per Criterion")
        ax.set_ylabel("Accuracy (|error| ≤ 1)")
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=20)

        self._annotate_bars(ax, adj_acc, fmt="{:.3f}", offset=0.01)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()


    def save_pearson_barplot(self, y_true: np.ndarray, y_pred: np.ndarray,
                            criteria: list[str], save_path: str) -> None:
        pears = []
        for i in range(len(criteria)):
            r, _ = pearsonr(y_true[:, i], y_pred[:, i])
            pears.append(r)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(criteria, pears)

        ax.set_title("Pearson Correlation per Criterion")
        ax.set_ylabel("Pearson r")
        ax.set_ylim(None, 1)
        ax.tick_params(axis='x', rotation=20)

        self._annotate_bars(ax, pears, fmt="{:.3f}", offset=0.02)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def save_spearman_barplot(self, y_true: np.ndarray, y_pred: np.ndarray,
                            criteria: list[str], save_path: str) -> None:
        spears = []
        for i in range(len(criteria)):
            r, _ = spearmanr(y_true[:, i], y_pred[:, i])
            spears.append(r)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(criteria, spears, color='teal')

        ax.set_title("Spearman Correlation per Criterion")
        ax.set_ylabel("Spearman ρ")
        ax.set_ylim(None, 1)
        ax.tick_params(axis='x', rotation=20)

        self._annotate_bars(ax, spears, fmt="{:.3f}", offset=0.02)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()



    # ===================
    # RUN FULL EVALUATION
    # ===================
    def run_full_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            criteria: list[str], bands: np.ndarray, 
                            save_dir: str, loss_history: list = [None]) -> None:
        """
        Runs the full evaluation pipeline and generates all evaluation plots.

        Arguments:
            y_true (np.ndarray): Ground truth scores with shape
                (num_samples, num_criteria).
            y_pred (np.ndarray): Predicted scores with shape
                (num_samples, num_criteria).
            criteria (list[str]): Names of evaluation criteria.
            bands (np.ndarray): Possible discrete band values.
            loss_history (list[float]): Training loss values for each epoch.
            save_dir (str): Directory where all evaluation results will be saved.

        Returns:
            None
        """
        y_true_float = np.asarray(y_true, dtype=float)
        y_pred_float = np.asarray(y_pred, dtype=float)        

        # y_true_int = self.snap_to_bands(y_true, bands, scale_for_int=True)  # for QWK/CM
        # y_pred_int = self.snap_to_bands(y_pred, bands, scale_for_int=True)
        y_true_int = np.round(y_true)
        y_pred_int = np.round(y_pred)

        # self.plot_loss_curve(loss_history, f"{save_dir}/loss_curve.png")
        self.save_mae_barplot(y_true_float, y_pred_float, criteria, f"{save_dir}/mae_per_criterion.png")
        self.save_mse_barplot(y_true_float, y_pred_float, criteria, f"{save_dir}/mse_per_criterion.png")
        self.save_qwk_barplot(y_true_int, y_pred_int, criteria, f"{save_dir}/qwk_per_criterion.png")
        self.save_error_histograms(y_true_float, y_pred_float, criteria, save_dir)

        self.save_rmse_barplot(y_true_float, y_pred_float, criteria, f"{save_dir}/rmse.png")
        self.save_adjacent_accuracy_barplot(y_true_float, y_pred_float, criteria, f"{save_dir}/adj_acc.png")
        self.save_pearson_barplot(y_true_float, y_pred_float, criteria, f"{save_dir}/pearson.png")
        self.save_spearman_barplot(y_true_float, y_pred_float, criteria, f"{save_dir}/spearman.png")

        self.save_calibration_curve_binned(y_true_float, y_pred_float, criteria, save_dir)
  
        # self.save_calibration_curve_binned(y_true_float, y_pred_float, criteria, save_dir)
        # for i, name in enumerate(criteria):
        #     self.save_confusion_matrix_with_original_labels(
        #         y_true_int[:,i],
        #         y_pred_int[:,i],
        #         y_true_float[:,i],
        #         name,
        #         f"{save_dir}/confusion_{name}.png"
        #     )
        # self.save_predictions_table(y_true_float, y_pred_float, criteria, f"{save_dir}/predictions_table.csv")

