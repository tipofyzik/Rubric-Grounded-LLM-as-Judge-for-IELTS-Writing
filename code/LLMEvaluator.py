import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from typing import Callable

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, spearmanr



class LLMEvaluator:
    """
    A class to evaluate the performance of LLMs across multiple runs and models.
    """
    def __init__(self, results_folder_path: str, y_true_path: str, dpi: int = 500):
        """
        Initialize the evaluator.

        Args:
            results_folder_path (str): Path to the folder containing result CSVs.
            y_true_path (str): Path to the CSV file containing true values.
            dpi (int): DPI for saved plots.
        """
        self.results_folder_path = results_folder_path
        self.y_true_df = pd.read_csv(y_true_path).sort_values("index").reset_index(drop=True)
        self.dpi = dpi

        self.true_names = {
            "non_fine_tuned_raw_result": "Zero-shot", 
            "non_fine_tuned_raw_rubric_result": "Rubric-based",
            "non_fine_tuned_raw_panda_result": "PANDA-based",
            "non_fine_tuned_raw_rubric_panda_result": "Rubric + PANDA-based",
            "non_fine_tuned_raw_panda_scores_result": "PANDA-based + perelimnary scores",
            "non_fine_tuned_raw_rubric_panda_scores_result": "Rubric + PANDA-based + perelimnary scores",
            "fine_tuned_raw_result": "Zero-shot (fine-tuned)", 
            "fine_tuned_raw_rubric_result": "Zero-shot (fine-tuned) + Rubric",
            "fine_tuned_raw_rubric_panda_result": "Zero-shot (fine-tuned) + Rubric + PANDA-based",
            "fine_tuned_raw_panda_scores_result": "Zero-shot (fine-tuned) + PANDA-based + perelimnary scores",
        }

        self.criteria = [
            "task_achievement",
            "coherence_and_cohesion",
            "lexical_resource",
            "grammatical_range_and_accuracy"
        ]
        self.true_criteria_names = {
            "task_achievement": "Task Achievement",
            "coherence_and_cohesion": "Coherence and Cohesion",
            "lexical_resource": "Lexical Resource",
            "grammatical_range_and_accuracy": "Grammatical Range and Accuracy"
        }
        self.true_model_names = {
            "mistral_7B": "Mistral-7B-Instruct-v0.2",
            "llama_32_3B": "Llama-3.2-3B-Instruct",
            "qwen25": "Qwen 2.5-1.5B-Instruct",
            "qwen2": "Qwen 2-1.5B-Instruct",
            "mistral_7B_4layers": "Mistral-7B-Instruct-v0.2 (fine-tuned)",
            "llama_32_3B_4layers": "Llama-3.2-3B-Instruct (fine-tuned)",
            "qwen25_4layers": "Qwen 2.5-1.5B-Instruct (fine-tuned)",
            "qwen2_4layers": "Qwen 2-1.5B-Instruct (fine-tuned)"
        }
        self.bands = np.arange(0, 10, 1)

    def load_all(self, run_order: list[str] | None = None) -> None:
        """
        Loads all CSV files and builds nested structure:
        self.all_data[file][model][run_id] -> DataFrame

        Args:
            run_order (list[str] | None): Optional explicit file ordering.

        Returns:
            None
        """
        self.all_data = {}
        self.run_order = run_order

        files = [f for f in os.listdir(self.results_folder_path) if f.endswith(".csv")]

        if run_order is not None:
            files = [f"{name}.csv" for name in run_order if f"{name}.csv" in files]
        else:
            files = sorted(files)

        for file in files:
            path = os.path.join(self.results_folder_path, file)
            df = pd.read_csv(path)

            file_key = file.replace(".csv", "")

            file_dict = {}

            for model, model_df in df.groupby("model"):

                run_dict = {}

                for run_id, run_df in model_df.groupby("file_id"):

                    run_df = (
                        run_df
                        .sort_values("sample_id")
                        .reset_index(drop=True)
                    )

                    run_dict[int(run_id)] = run_df  # type: ignore

                file_dict[model] = run_dict

            self.all_data[file_key] = file_dict

    def _get_all_models(self) -> list[str]:
        """
        Collects all unique model names across loaded datasets.

        Returns:
            list[str]: Sorted list of model names.
        """
        models = set()
        for file_data in self.all_data.values():
            models.update(file_data.keys())
        return sorted(models)

    def _get_true_pred(self, run_df: pd.DataFrame, crit: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Aligns true and predicted values for a given criterion.

        Args:
            run_df (pd.DataFrame): Prediction dataframe.
            crit (str): Evaluation criterion.

        Returns:
            tuple[np.ndarray, np.ndarray]: (y_true, y_pred)
        """
        y_true = self.y_true_df[crit].to_numpy()
        y_pred = run_df[crit].to_numpy()

        n = min(len(y_true), len(y_pred))

        return y_true[:n], y_pred[:n]
    
    def _aggregate_over_runs(self, runs: dict, crit: str, function: Callable) -> tuple[float, float]:
        """
        Aggregates metric over multiple runs.

        Args:
            runs (dict): run_id -> DataFrame.
            crit (str): evaluation criterion.
            function (callable): function(y_true, y_pred) -> float.

        Returns:
            tuple[float, float]: (mean, std)
        """
        values = []

        for run_df in runs.values():
            y_true, y_pred = self._get_true_pred(run_df, crit)
            values.append(function(y_true, y_pred))

        values = np.array(values)

        return values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0

    def _qwk(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes Quadratic Weighted Kappa.

        Returns:
            float: QWK score (0.0 if undefined)
        """
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return 0.0 if np.isnan(qwk) else qwk

    def _mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.floating:
        """
        Computes Mean Absolute Error.

        Returns:
            float: MAE value
        """
        return np.mean(np.abs(y_true - y_pred))

    def _rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes Root Mean Squared Error.

        Returns:
            float: RMSE value
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def _bias_fn(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes prediction bias.

        Returns:
            np.ndarray: (y_pred - y_true)
        """
        return y_pred - y_true
    
    def _adjacent_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred) <= 0.5))
    
    def _pearson_corr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0
        return pearsonr(y_true, y_pred)[0]  #type: ignore

    def _spearman_corr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0
        return spearmanr(y_true, y_pred)[0] #type: ignore

    def _calibration_fn(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Returns raw predictions for calibration analysis.

        Returns:
            np.ndarray: y_pred
        """
        return y_pred

    def _plot_bar_metric(self, metric_function: Callable, ylabel: str, 
                         title_fmt: str, save_name: str, 
                         save_dir: str | None = None, 
                         y_lim: tuple[float|None, float|None] | None = None) -> None:
        """
        Plots bar metrics across criteria and datasets.

        Args:
            metric_fn (callable): metric function (y_true, y_pred)
            ylabel (str): y-axis label
            title_fmt (str): title format string
            save_name (str): output filename suffix
            save_dir (str | None): save directory
            y_lim (tuple[float, float] | None): optional y-axis limits

        Returns:
            None
        """
        if not self.all_data:
            raise ValueError("self.all_data is empty. Did you call load_all()?")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        file_keys = list(self.all_data.keys())
        if self.run_order is not None:
            file_keys = [k for k in self.run_order if k in file_keys]

        models = self._get_all_models()
        for model in models:
            plt.figure(figsize=(12, 6))

            x = np.arange(len(self.criteria))
            width = 0.8 / max(len(file_keys), 1)

            all_mean_values = []

            for i, file_key in enumerate(file_keys):

                file_data = self.all_data[file_key]

                if model not in file_data:
                    continue

                runs = file_data[model]

                mean_per_crit = []
                std_per_crit = []

                for crit in self.criteria:

                    run_values = []

                    for run_df in runs.values():

                        y_true = self.y_true_df[crit].to_numpy()
                        y_pred = run_df[crit].to_numpy()

                        n = min(len(y_true), len(y_pred))

                        y_true = y_true[:n]
                        y_pred = y_pred[:n]

                        val = metric_function(y_true, y_pred)
                        run_values.append(val)

                    run_values = np.array(run_values)

                    mean_per_crit.append(run_values.mean())
                    all_mean_values.append(run_values.mean())
                    std_per_crit.append(run_values.std(ddof=1) if len(run_values) > 1 else 0.0)

                bars = plt.bar(
                    x + i * width,
                    mean_per_crit,
                    width=width,
                    yerr=std_per_crit,
                    capsize=4,
                    label=self.true_names[file_key]
                )

                # value labels
                for bar, val in zip(bars, mean_per_crit):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.025,
                        f"{val:.3f}",
                        ha="center",
                        fontsize=8
                    )
                
            
            plt.xticks(
                x + width * (len(file_keys) - 1) / 2,
                [self.true_criteria_names[crit] for crit in self.criteria]
                # rotation=45
            )

            if y_lim is not None:
                plt.ylim(*y_lim)
            else:
                plt.ylim(0, max(all_mean_values) + 0.225)

            plt.ylabel(ylabel)
            plt.title(title_fmt.format(model=self.true_model_names[model]))

            plt.grid(axis="y")

            plt.legend(
                title="Prompt configurations",
                loc="lower center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=min(len(file_keys), 3)
            )
            plt.tight_layout()

            if save_dir:
                safe_model = model.replace("/", "_")
                plt.savefig(
                    os.path.join(save_dir, f"{safe_model}_{save_name}.{self.image_format}"),
                    format=self.image_format,
                    bbox_inches="tight",
                    dpi=self.dpi
                )

            plt.close()



    def _plot_bar_metric_all_models(
        self,
        metric_function: Callable,
        ylabel: str,
        title_fmt: str,
        save_name: str,
        save_dir: str | None = None,
        y_lim: tuple[float | None, float | None] | None = None,
    ) -> None:

        if not self.all_data:
            raise ValueError("self.all_data is empty. Did you call load_all()?")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        file_keys = list(self.all_data.keys())

        if self.run_order is not None:
            file_keys = [k for k in self.run_order if k in file_keys]

        models = self._get_all_models()

        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        model_colors = {
            model: default_colors[i % len(default_colors)]
            for i, model in enumerate(models)
        }

        # =========================================================
        # FIGURE 2x2 (criteria)
        # =========================================================
        fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=False)
        axes = axes.flatten()

        all_values = []
        csv_rows = []

        section_gap = 2.0

        # =========================================================
        # BUILD X POSITIONS PER CRITERION
        # =========================================================
        section_positions = []
        current_x = 0

        for _ in self.criteria:
            xs = np.arange(len(file_keys)) + current_x
            section_positions.append(xs)
            current_x += len(file_keys) + section_gap

        # =========================================================
        # MOVING AVERAGE
        # =========================================================
        def moving_average(y, window=3):
            y = np.asarray(y, dtype=float)
            out = np.full_like(y, np.nan, dtype=float)

            for i in range(len(y)):
                left = max(0, i - window // 2)
                right = min(len(y), i + window // 2 + 1)

                chunk = y[left:right]
                chunk = chunk[~np.isnan(chunk)]

                if len(chunk) > 0:
                    out[i] = chunk.mean()

            return out

        # =========================================================
        # BUILD CSV TABLE
        # =========================================================
        for model in models:

            # строка с названием модели
            csv_rows.append(
                [self.true_model_names.get(model, model)]
                + [""] * len(self.criteria)
            )

            # header
            crits = ["TA", "CC", "LR", "GRA"]
            csv_rows.append(
                ["Prompt"]
                + [crits[i] for i in range(len(crits))]
            )

            for i, file_key in enumerate(file_keys):

                row = [f"p{i+1}"]

                file_data = self.all_data[file_key]

                if model not in file_data:
                    row.extend([""] * len(self.criteria))
                    csv_rows.append(row)
                    continue

                runs = file_data[model]

                for crit in self.criteria:

                    run_values = []

                    for run_df in runs.values():

                        y_true, y_pred = self._get_true_pred(run_df, crit)

                        run_values.append(
                            metric_function(y_true, y_pred)
                        )

                    run_values = np.array(run_values)

                    mean = run_values.mean()
                    std = run_values.std(ddof=1) if len(run_values) > 1 else 0.0

                    row.append(f"{mean:.3f} ± {std:.3f}")

                csv_rows.append(row)

            # пустая строка между моделями
            csv_rows.append([""] * (len(self.criteria) + 1))

        # =========================================================
        # LOOP OVER CRITERIA (each subplot)
        # =========================================================
        for crit_idx, crit in enumerate(self.criteria):

            ax = axes[crit_idx]
            xs = section_positions[crit_idx]

            for model in models:

                color = model_colors[model]
                means = []

                for i, file_key in enumerate(file_keys):

                    file_data = self.all_data[file_key]

                    if model not in file_data:
                        means.append(np.nan)
                        continue

                    runs = file_data[model]

                    run_values = []
                    for run_df in runs.values():
                        y_true, y_pred = self._get_true_pred(run_df, crit)
                        run_values.append(metric_function(y_true, y_pred))

                    run_values = np.array(run_values)

                    mean = run_values.mean()
                    means.append(mean)
                    all_values.append(mean)

                    std = run_values.std(ddof=1) if len(run_values) > 1 else 0.0

                    ax.scatter(
                        xs[i],
                        mean,
                        s=18,
                        color=color,
                        edgecolor="black",
                        linewidth=0.8,
                        zorder=3
                    )

                    ax.errorbar(
                        xs[i],
                        mean,
                        yerr=std,
                        fmt="none",
                        color=color,
                        capsize=4,
                        alpha=0.95,
                        zorder=2
                    )

                # =====================================================
                # LINES
                # =====================================================
                xs_arr = np.array(xs)
                means_arr = np.array(means)

                valid = ~np.isnan(means_arr)
                xs_valid = xs_arr[valid]
                means_valid = means_arr[valid]

                if len(xs_valid) >= 2:
                    ax.plot(
                        xs_valid,
                        means_valid,
                        color=color,
                        linewidth=1.25,
                        alpha=0.95,
                        linestyle="-",
                        zorder=2
                    )

                # if len(means_valid) >= 3:
                #     ma = moving_average(means_valid)

                #     ax.plot(
                #         xs_valid,
                #         ma,
                #         color=color,
                #         linewidth=1.25,
                #         alpha=0.65,
                #         linestyle="--",
                #         zorder=2
                #     )

            # =========================================================
            # GRID (stronger)
            # =========================================================
            ax.grid(True, axis="both", alpha=0.5, linewidth=0.8)

            # =========================================================
            # INTERNAL SEPARATORS
            # =========================================================
            for i in range(len(xs) - 1):
                mid = (xs[i] + xs[i + 1]) / 2
                ax.axvline(mid, linestyle=":", color="black", alpha=0.25)

            # =========================================================
            # OUTER BORDERS (NEW)
            # =========================================================
            ax.axvline(xs[0] - 0.5, linestyle=":", color="black", alpha=0.25)
            ax.axvline(xs[-1] + 0.5, linestyle=":", color="black", alpha=0.25)

            # =========================================================
            # PROMPT LABELS
            # =========================================================
            prompt_labels = [f"p{i+1}" for i in range(len(file_keys))]

            for i, x in enumerate(xs):
                ax.text(
                    x,
                    -0.01,
                    prompt_labels[i],
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=12,
                    alpha=0.9
                )

            ax.set_title(self.true_criteria_names[crit], fontweight="bold")
            ax.set_xticks([])

        # =========================================================
        # GLOBAL LEGEND (MODELS)
        # =========================================================
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        model_handles = [
            Patch(
                facecolor=model_colors[m],
                edgecolor="black",
                label=self.true_model_names.get(m, m)
            )
            for m in models
        ]
        line_handles = [
            Line2D(
                [0], [0],
                color="black",
                linestyle="-",
                linewidth=1.5,
                label="Raw trend"
            ),
            Line2D(
                [0], [0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Moving average"
            )
        ]
        prompt_handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                color="black",
                label=f"p{i+1}: {self.true_names[file_keys[i]]}"
            )
            for i in range(len(file_keys))
        ]

        # fig.legend(
        #     handles=model_handles + line_handles,
        #     title="Models & trend type",
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, 1.06),
        #     ncol=min(len(models) + 2, 6)
        # )
        fig.legend(
            handles=model_handles,
            title="Models",
            loc="upper right",
            bbox_to_anchor=(0.4 if len(file_keys) == 6 else 0.45, 1.0),
            ncol=min(len(models) + 2, 2),
            fontsize = 12
        )
        fig.legend(
            handles=prompt_handles,
            title="Prompting configurations",
            loc="upper left",
            bbox_to_anchor=(0.4 if len(file_keys) == 6 else 0.45, 1.0),
            ncol=3 if len(file_keys) == 6 else 2,
            fontsize = 12
        )
        

        # =========================================================
        # Y LIMITS
        # =========================================================
        if y_lim is not None:
            ymin, ymax = y_lim
            if ymin is None:
                ymin = min(all_values) - 0.05
            if ymax is None:
                ymax = max(all_values) + 0.05
            for ax in axes:
                ax.set_ylim(ymin, ymax)
        else:
            lim = (min(all_values) - 0.05, max(all_values) + 0.10)
            for ax in axes:
                ax.set_ylim(*lim)

        # =========================================================
        # FINAL COSMETICS
        # =========================================================
        for ax in axes:
            ax.set_ylabel(ylabel, fontsize = 12)
            ax.axhline(0, linestyle="--", linewidth=1.5, color="black", alpha=0.7, zorder=1)
            ax.set_xlabel("Prompting configuration", labelpad=20, fontsize = 12)

        fig.suptitle(title_fmt, fontsize=16, y=0.9)
        plt.tight_layout(rect=(0, 0, 1, 0.90))

        # =========================================================
        # SAVE
        # =========================================================
        if save_dir:
            plt.savefig(
                os.path.join(save_dir, f"{save_name}.{self.image_format}"),
                format=self.image_format,
                bbox_inches="tight",
                dpi=self.dpi
            )
        # =========================================================
        # SAVE CSV
        # =========================================================
        if save_dir:

            csv_df = pd.DataFrame(csv_rows)

            csv_df.to_csv(
                os.path.join(save_dir, f"{save_name}.csv"),
                index=False,
                header=False
            )
        plt.close()







    def _binned_curve_over_runs(self, runs: dict, crit: str, 
                                bins: np.ndarray, value_fn: Callable) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes per-bin statistic curves across runs.

        Args:
            runs (dict): run_id -> DataFrame
            crit (str): evaluation criterion
            bins (np.ndarray): y_true bins
            value_fn (callable): function(y_true, y_pred) -> values

        Returns:
            tuple[np.ndarray, np.ndarray]: (mean_curve, std_curve)
        """

        all_curves = []

        for run_df in runs.values():

            y_true = self.y_true_df[crit].to_numpy()
            y_pred = run_df[crit].to_numpy()

            n = min(len(y_true), len(y_pred))

            y_true = np.round(y_true[:n])
            y_pred = y_pred[:n]

            curve = []

            for v in bins:
                mask = y_true == v

                if np.sum(mask) == 0:
                    curve.append(0.0)
                else:
                    curve.append(
                        value_fn(y_true[mask], y_pred[mask]).mean()
                    )

            all_curves.append(curve)

        all_curves = np.array(all_curves)

        mean = all_curves.mean(axis=0)
        std = all_curves.std(axis=0, ddof=1) if len(all_curves) > 1 else np.zeros_like(mean)

        return mean, std

    def _plot_binned_metric(self, metric_type: str, value_fn: Callable, 
                            ylabel: str, title_fmt: str, save_name: str, 
                            save_dir: str | None = None, mode: str = "single",) -> None:
        """
        Plots binned metrics across runs.

        Args:
            metric_type (str): "bias" | "calibration"
            value_fn (callable): per-sample transformation
            ylabel (str): y-axis label
            title_fmt (str): title format string
            save_name (str): output filename suffix
            save_dir (str | None): save directory
            mode (str): "grid" | "single"

        Returns:
            None
        """
        if not self.all_data:
            raise ValueError("self.all_data is empty. Did you call load_all()?")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        file_keys = list(self.all_data.keys())
        if self.run_order is not None:
            file_keys = [k for k in self.run_order if k in file_keys]

        models = self._get_all_models()

        if metric_type not in {"bias", "calibration"}:
            raise ValueError("metric_type must be 'bias' or 'calibration'")

        def add_reference_line(ax):
            if metric_type == "bias":
                ax.axhline(0, linestyle="--", color="black")
                ax.set_xticks(np.arange(0, 10, 1))
            elif metric_type == "calibration":
                ax.plot([0, 9], [0, 9], "--", color="black")
                ax.set_xticks(np.arange(0, 10, 1))
                ax.set_yticks(np.arange(0, 10, 1))

        for model in models:

            # =========================
            # GRID
            # =========================
            if mode == "grid":

                fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
                axes = axes.flatten()

                for ax, crit in zip(axes, self.criteria):

                    y_true_full = np.round(self.y_true_df[crit].to_numpy())
                    bins = np.unique(y_true_full)

                    for file_key in file_keys:

                        file_data = self.all_data[file_key]
                        if model not in file_data:
                            continue

                        runs = file_data[model]

                        mean_curve, std_curve = self._binned_curve_over_runs(
                            runs, crit, bins, value_fn
                        )

                        ax.errorbar(
                            bins,
                            mean_curve,
                            yerr=std_curve,
                            marker="o",
                            capsize=3,
                            label=self.true_names[file_key]
                        )

                    add_reference_line(ax)
                    ax.set_title(self.true_criteria_names[crit])
                    ax.grid()
                    ax.tick_params(labelbottom=True, labelleft=True)

                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.96),
                    ncol=min(len(file_keys), 3),
                    title="Prompt configurations"
                )

                fig.suptitle(f"{self.true_model_names[model]} | {metric_type} (grid)", fontsize=14, y=0.95)
                fig.supxlabel("y_true")
                fig.supylabel(ylabel)

                plt.tight_layout(rect=(0, 0, 0.9, 0.95))

                if save_dir:
                    safe_model = model.replace("/", "_")
                    plt.savefig(
                        os.path.join(save_dir, f"{safe_model}_{save_name}_grid.{self.image_format}"),
                        dpi=self.dpi,
                        bbox_inches="tight"
                    )

                plt.close()

            # =========================
            # SINGLE
            # =========================
            elif mode == "single":

                for crit in self.criteria:

                    plt.figure(figsize=(10, 6))

                    y_true_full = self.y_true_df[crit].to_numpy()
                    bins = np.unique(np.round(y_true_full))

                    for file_key in file_keys:

                        file_data = self.all_data[file_key]
                        if model not in file_data:
                            continue

                        runs = file_data[model]

                        mean_curve, std_curve = self._binned_curve_over_runs(
                            runs, crit, bins, value_fn
                        )

                        plt.errorbar(
                            bins,
                            mean_curve,
                            yerr=std_curve,
                            marker="o",
                            capsize=3,
                            label=self.true_names[file_key]
                        )

                    # reference line
                    if metric_type == "bias":
                        plt.axhline(0, linestyle="--", color="black")
                    else:  # calibration
                        plt.plot([0, 9], [0, 9], "--", color="black")
                        plt.yticks(np.arange(0, 10, 1))

                    plt.xticks(np.arange(0, 10, 1))
                    plt.xlabel("y_true")
                    plt.ylabel(ylabel)
                    plt.title(title_fmt.format(model=self.true_model_names[model], crit=crit, crit_name=self.true_criteria_names[crit]))

                    plt.grid()
                    plt.legend(
                        title="Prompt configurations",
                        loc="lower center",
                        bbox_to_anchor=(0.5, 1.15),
                        ncol=min(len(file_keys), 3)
                    )
                    plt.tight_layout()

                    if save_dir:
                        safe_model = model.replace("/", "_")
                        plt.savefig(
                            os.path.join(save_dir, f"{safe_model}_{crit}_{save_name}_single.{self.image_format}"),
                            dpi=self.dpi,
                            bbox_inches="tight"
                        )

                    plt.close()

    def _error_hist_over_runs(self, runs: dict, crit: str, 
                              bins: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes error histograms across multiple runs.

        For each run:
            - Aligns y_true and y_pred
            - Computes per-sample error: (y_pred - y_true)
            - Builds histogram of errors using shared binning

        Across runs:
            - Stacks histogram counts into a matrix

        Args:
            runs (dict): Mapping run_id -> DataFrame with predictions
            crit (str): Evaluation criterion column name
            bins (int): Number of histogram bins

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - all_run_counts: shape (num_runs, num_bins)
                Histogram counts per run
                - bin_edges: shared histogram bin edges
        """
        all_run_counts = []
        bin_edges = None

        for run_df in runs.values():

            y_true = self.y_true_df[crit].to_numpy()
            y_pred = run_df[crit].to_numpy()

            n = min(len(y_true), len(y_pred))
            y_true = y_true[:n]
            y_pred = y_pred[:n]

            errors = y_pred - y_true

            counts, bin_edges = np.histogram(errors, bins=bins)
            all_run_counts.append(counts)

        if bin_edges is None:
            return np.empty((0, 0)), np.array([])
        return np.array(all_run_counts), bin_edges

    def _plot_error_hist_metric(
        self,
        metric_type: str,   # "mean" | "std"
        bins: int = 25,
        save_dir: str | None = None,
        mode: str = "single",  # "single" | "grid"
        smooth: str | None = None,  # None, "gaussian", "spline"
    ) -> None:
        """
        Plots error histograms aggregated across multiple runs for each model
        and evaluation criterion, comparing different prompt configurations.

        --------------------------------------------------------------------
        WHAT IS COMPUTED
        --------------------------------------------------------------------

        For each file and model:
        - Compute prediction error: error = y_pred - y_true
        - Build histogram per run over fixed bins
        - Stack histograms across runs → (num_runs, num_bins)

        --------------------------------------------------------------------
        metric_type = "mean"
        --------------------------------------------------------------------
        For each bin:
            mean_counts = average bin count across runs

        Shows:
        - Typical / expected error distribution
        - Stable, averaged failure profile of the model

        --------------------------------------------------------------------
        metric_type = "std"
        --------------------------------------------------------------------
        For each bin:
            std_counts = standard deviation of bin counts across runs

        Shows:
        - Stability of the error distribution
        - How much the model's failure modes vary across runs

        High values indicate:
        - unstable error patterns
        - sensitivity to sampling / decoding randomness

        Low values indicate:
        - consistent and reproducible error behavior

        --------------------------------------------------------------------
        OUTPUT
        --------------------------------------------------------------------

        Line plot where:
        - x-axis: error bins (y_pred - y_true)
        - y-axis: mean or std of counts across runs
        - each line: one prompt configuration
        """
        
        if not self.all_data:
            raise ValueError("self.all_data is empty. Did you call load_all()?")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        file_keys = list(self.all_data.keys())
        if self.run_order is not None:
            file_keys = [k for k in self.run_order if k in file_keys]

        models = self._get_all_models()

        if metric_type not in {"mean", "std"}:
            raise ValueError("metric_type must be 'mean' or 'std'")

        if mode not in {"single", "grid", "ridge"}:
            raise ValueError("mode must be 'single' or 'grid' or 'ridge'")

        if smooth not in {"None", "gaussian", "spline"}:
            raise ValueError("smooth must be 'None', 'gaussian' or 'spline'")


        def agg_fn(x: np.ndarray) -> np.ndarray:
            return x.mean(axis=0) if metric_type == "mean" else x.std(axis=0, ddof=1)

        def plot_line(ax, x: np.ndarray, y: np.ndarray, label: str):
            if smooth == "gaussian":
                y_smooth = gaussian_filter1d(y, sigma=1.0)
                ax.plot(x, y_smooth, linewidth=2, label=label)
                ax.scatter(x, y, s=15)

            elif smooth == "spline" and len(x) >= 4:
                x_s = np.linspace(x.min(), x.max(), 300)
                y_s = make_interp_spline(x, y)(x_s)
                ax.plot(x_s, y_s, linewidth=2, label=label)
                ax.scatter(x, y, s=15)

            else:
                ax.plot(x, y, marker="o", linewidth=2, label=label)

        ylabel = (
            "Mean number of samples per bin"
            if metric_type == "mean"
            else "STD of samples per bin across runs"
        )
        title_suffix = "mean histogram" if metric_type == "mean" else "std histogram"

        # =========================
        # GRID MODE
        # =========================
        if mode == "grid":

            for model in models:

                fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
                axes = axes.flatten()

                for ax, crit in zip(axes, self.criteria):

                    found_any = False

                    for file_key in file_keys:

                        file_data = self.all_data[file_key]
                        if model not in file_data:
                            continue

                        runs = file_data[model]

                        all_run_counts, bin_edges = self._error_hist_over_runs(
                            runs, crit, bins
                        )

                        if all_run_counts.size == 0 or bin_edges.size == 0:
                            continue

                        values = agg_fn(all_run_counts)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                        plot_line(ax, bin_centers, values, self.true_names[file_key])

                        found_any = True

                    if found_any:
                        ax.axvline(0, linestyle="--", color="black")
                        ax.set_title(self.true_criteria_names[crit])
                        ax.grid()
                        ax.tick_params(labelbottom=True, labelleft=True)

                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.96),
                    ncol=min(len(file_keys), 3),
                    title="Prompt configurations"
                )

                fig.suptitle(f"{self.true_model_names[model]} | error histogram ({title_suffix})", fontsize=14, y=0.95)
                fig.supxlabel("y_pred - y_true")
                fig.supylabel(ylabel)
                plt.tight_layout(rect=(0, 0, 0.9, 0.95))

                if save_dir:
                    safe_model = model.replace("/", "_")
                    plt.savefig(
                        os.path.join(
                            save_dir,
                            f"{safe_model}_error_hist_{metric_type}_grid.{self.image_format}"
                        ),
                        dpi=self.dpi,
                        bbox_inches="tight",
                        format=self.image_format
                    )

                plt.close()

        # =========================
        # SINGLE MODE
        # =========================
        elif mode == "single":

            for model in models:

                for crit in self.criteria:

                    fig, ax = plt.subplots(figsize=(10, 6))

                    found_any = False

                    for file_key in file_keys:

                        file_data = self.all_data[file_key]
                        if model not in file_data:
                            continue

                        runs = file_data[model]

                        all_run_counts, bin_edges = self._error_hist_over_runs(
                            runs, crit, bins
                        )

                        if all_run_counts.size == 0 or bin_edges.size == 0:
                            continue

                        values = agg_fn(all_run_counts)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                        plot_line(ax, bin_centers, values, self.true_names[file_key])

                        found_any = True

                    if not found_any:
                        plt.close()
                        continue

                    ax.axvline(0, linestyle="--", color="black")

                    ax.set_xlabel("y_pred - y_true")
                    ax.set_ylabel(ylabel)
                    ax.set_title(f"{self.true_model_names[model]} | {self.true_criteria_names[crit]} | error histogram ({title_suffix})")

                    ax.grid()
                    ax.legend(title="Prompt configurations")

                    if save_dir:
                        safe_model = model.replace("/", "_")
                        plt.savefig(
                            os.path.join(
                                save_dir,
                                f"{safe_model}_{self.true_criteria_names[crit]}_error_hist_{metric_type}_across_files.{self.image_format}"
                            ),
                            bbox_inches="tight",
                            dpi=self.dpi,
                            format=self.image_format
                        )

                    plt.close()

        elif mode == "ridge":

            for model in models:

                for crit in self.criteria:

                    fig, ax = plt.subplots(figsize=(10, 6))

                    curves = []

                    # =========================
                    # собираем данные сначала
                    # =========================
                    for file_key in file_keys:

                        file_data = self.all_data[file_key]
                        if model not in file_data:
                            continue

                        runs = file_data[model]

                        all_run_counts, bin_edges = self._error_hist_over_runs(
                            runs, crit, bins
                        )

                        if all_run_counts.size == 0 or bin_edges.size == 0:
                            continue

                        values = agg_fn(all_run_counts)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                        # smoothing
                        if smooth == "gaussian":
                            values = gaussian_filter1d(values, sigma=1.0)

                        elif smooth == "spline" and len(bin_centers) >= 4:
                            x_s = np.linspace(bin_centers.min(), bin_centers.max(), 300)
                            values = make_interp_spline(bin_centers, values)(x_s)
                            bin_centers = x_s

                        curves.append((file_key, bin_centers, values))

                    if not curves:
                        plt.close()
                        continue

                    # =========================
                    # динамический offset
                    # =========================
                    global_max = max(v.max() for _, _, v in curves)
                    step = global_max * 1.3 if global_max > 0 else 1.0

                    # =========================
                    # рисуем
                    # =========================
                    yticks = []

                    for i, (file_key, x, y) in enumerate(curves):

                        offset = i * step
                        y_shifted = y + offset

                        ax.plot(x, y_shifted, linewidth=2, label=self.true_names[file_key])
                        ax.fill_between(x, offset, y_shifted, alpha=0.2)

                        # центр распределения (без нормализации!)
                        if y.sum() > 0:
                            center = np.sum(x * y) / np.sum(y)
                            ax.axvline(center, linestyle=":", alpha=0.4)

                        yticks.append(offset)

                        # разделительная линия
                        ax.axhline(offset, linestyle="--", alpha=0.15)

                    # =========================
                    # оси
                    # =========================
                    ax.set_yticks(yticks)
                    ax.set_ylabel(ylabel)

                    ax.axvline(0, linestyle="--", color="black")

                    ax.set_xlabel("y_pred - y_true")
                    ax.set_title(f"{self.true_model_names[model]} | {self.true_criteria_names[crit]} | error ridgeline ({title_suffix})")

                    ax.grid(axis="x")
                    ax.legend(
                        title="Prompt configurations",
                        loc="lower center",
                        bbox_to_anchor=(0.5, 1.15),
                        ncol=min(len(file_keys), 3)
                    )



                    if save_dir:
                        safe_model = model.replace("/", "_")
                        plt.savefig(
                            os.path.join(
                                save_dir,
                                f"{safe_model}_{self.true_criteria_names[crit]}_error_hist_{metric_type}_ridge.{self.image_format}"
                            ),
                            bbox_inches="tight",
                            format=self.image_format,
                            dpi=self.dpi
                        )

                    plt.close()


    def plot_metrics(self, save_dir: str) -> None:
        """
        Runs full evaluation pipeline and saves all plots.

        Produces:
            - QWK / RMSE / MAE bar plots
            - bias & calibration plots
            - error histogram mean plots
            - error histogram std plots

        Args:
            save_dir (str): output directory

        Returns:
            None
        """     
        self.image_format = "webp" # "png" | "webp"

        self._plot_bar_metric(
            metric_function=self._qwk,
            ylabel="Mean QWK",
            title_fmt="{model} | Mean QWK across prompts",
            save_name="mean_qwk_across_runs",
            save_dir=f"{save_dir}/mean_qwk_across_runs",
            y_lim = (None, 1)
        )
        self._plot_bar_metric(
            metric_function=self._mae,
            ylabel="Mean MAE",
            title_fmt="{model} | Mean MAE across prompts",
            save_name="mean_mae_across_runs",
            save_dir=f"{save_dir}/mean_mae_across_runs"
        )
        self._plot_bar_metric(
            metric_function=self._rmse,
            ylabel="Mean RMSE",
            title_fmt="{model} | Mean RMSE across prompts",
            save_name="mean_rmse_across_runs",
            save_dir=f"{save_dir}/mean_rmse_across_runs",
        )
        self._plot_bar_metric(
            metric_function=self._adjacent_accuracy,
            ylabel="Adjacent Accuracy",
            title_fmt="{model} | Adjacent Accuracy ±0.5 score across prompts",
            save_name="adjacent_accuracy_across_runs",
            save_dir=f"{save_dir}/adjacent_accuracy_across_runs",
            y_lim = (0, 1)
        )
        self._plot_bar_metric(
            metric_function=self._pearson_corr,
            ylabel="Mean Pearson Correlation",
            title_fmt="{model} | Mean Pearson Correlation across prompts",
            save_name="mean_pearson_corr_across_runs",
            save_dir=f"{save_dir}/mean_pearson_corr_across_runs",
            y_lim = (None, 1)
        )
        self._plot_bar_metric(
            metric_function=self._spearman_corr,
            ylabel="Mean Spearman Correlation",
            title_fmt="{model} | Mean Spearman Correlation across prompts",
            save_name="mean_spearman_corr_across_runs",
            save_dir=f"{save_dir}/mean_spearman_corr_across_runs",
            y_lim = (None, 1)
        )



        self._plot_bar_metric_all_models(
            metric_function=self._qwk,
            ylabel="Mean QWK",
            title_fmt="Mean QWK across prompts",
            save_name="mean_qwk_across_runs",
            save_dir=f"{save_dir}/mean_metrics_across_runs",
            y_lim = (None, 0.7)
        )
        self._plot_bar_metric_all_models(
            metric_function=self._mae,
            ylabel="Mean MAE",
            title_fmt="Mean MAE across prompts",
            save_name="mean_mae_across_runs",
            save_dir=f"{save_dir}/mean_metrics_across_runs"
        )
        self._plot_bar_metric_all_models(
            metric_function=self._rmse,
            ylabel="Mean RMSE",
            title_fmt="Mean RMSE across prompts",
            save_name="mean_rmse_across_runs",
            save_dir=f"{save_dir}/mean_metrics_across_runs",
        )
        self._plot_bar_metric_all_models(
            metric_function=self._adjacent_accuracy,
            ylabel="Adjacent Accuracy",
            title_fmt="Adjacent Accuracy (±0.5 score points) across prompts",
            save_name="adjacent_accuracy_across_runs",
            save_dir=f"{save_dir}/mean_metrics_across_runs",
            y_lim = (0, 0.65)
        )
        self._plot_bar_metric_all_models(
            metric_function=self._pearson_corr,
            ylabel="Mean Pearson Correlation",
            title_fmt="Mean Pearson Correlation across prompts",
            save_name="mean_pearson_corr_across_runs",
            save_dir=f"{save_dir}/mean_metrics_across_runs",
            y_lim = (None, 0.75)
        )
        self._plot_bar_metric_all_models(
            metric_function=self._spearman_corr,
            ylabel="Mean Spearman Correlation",
            title_fmt="Mean Spearman Correlation across prompts",
            save_name="mean_spearman_corr_across_runs",
            save_dir=f"{save_dir}/mean_metrics_across_runs",
            y_lim = (None, 0.75)
        )



        mode = "grid" # "single" | "grid"
        # self._plot_binned_metric(
        #     metric_type = "bias",
        #     value_fn=self._bias_fn,
        #     ylabel="Mean bias (y_pred - y_true)",
        #     title_fmt="{model} | {crit_name} | Mean Bias across runs",
        #     save_name="bias",
        #     save_dir=f"{save_dir}/bias",
        #     mode=mode
        # )

        self._plot_binned_metric(
            metric_type = "calibration",
            value_fn=self._calibration_fn,
            ylabel="Mean y_pred",
            title_fmt="{model} | {crit_name} | Mean Calibration across runs",
            save_name="calibration",
            save_dir=f"{save_dir}/calibration",
            mode=mode   
        )

        # self._plot_error_hist_metric(
        #     metric_type="mean", 
        #     bins=25, 
        #     save_dir=f"{save_dir}/error_lines",
        #     mode=mode,
        #     smooth="None" # None, "gaussian", "spline"
        #     )
        # self._plot_error_hist_metric(
        #     metric_type="std", 
        #     bins=25, 
        #     save_dir=f"{save_dir}/error_std_lines",
        #     mode=mode,
        #     smooth="None" # None, "gaussian", "spline"
        #     )
    
