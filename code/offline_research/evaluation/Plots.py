"""Configurable Matplotlib reports over saved statistics; no smoothing."""
from typing import Any, Iterator

from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from offline_research.evaluation.Metrics import METRICS
from offline_research.evaluation.Report import AES_KEY, CRITERIA, SHORT_NAMES


_METRIC_RANGES = {
    "mae": (0, 9), "mse": (0, 81), "rmse": (0, 9),
    "qwk": (-1, 1), "pearson": (-1, 1), "spearman": (-1, 1),
    "exact_accuracy": (0, 1), "within_one_accuracy": (0, 1),
    "bias": (-9, 9), "calibration": (0, 9),
}


def _axis_limits(values: Any, metric: Any) -> Any:
    """Leave 5% of the data span above and below the plotted mean points.

    Args:
        values (Any): Caller-supplied settings or values used by this operation.
        metric (Any): Metric used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    floor, ceiling = _METRIC_RANGES[metric]
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return floor, ceiling
    minimum, maximum = values.min(), values.max()
    span = maximum - minimum
    padding = (span if span else abs(minimum) or ceiling - floor) * 0.05
    return minimum - padding, maximum + padding


def render_plots(tables: Any, plan: Any, output: Any, options: Any) -> dict[str, Any]:
    """Render plots.

    Args:
        tables (Any): Tables used by this operation.
        plan (Any): Plan used by this operation.
        output (Any): Output used by this operation.
        options (Any): Options used by this operation.

    Returns:
        dict[str, Any]: Named result fields and metadata produced by this operation.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    saved, skipped = [], []
    labels = {AES_KEY: "AES transformer (TR proxy)", **plan.get("model_labels", {})}
    setups = plan["setups"]
    palette = plt.get_cmap("tab10")
    series = ([(AES_KEY, "aes")] if plan.get("include_aes", True) else []) + [(current_match, current_text) for current_match in plan["models"] for current_text in setups]
    summary = tables["metrics_summary"]
    common = summary.loc[summary.population.eq("common_valid")]
    n_common = len(tables["common_ids"])
    style = options["style"]
    criterion_groups = [CRITERIA] if options["layout"] == "grid" else [(current_criterion,) for current_criterion in CRITERIA]

    def name(model: Any, setup: str) -> Any:
        """Name.

        Args:
            model (Any): Model used by this operation.
            setup (str): Context setup selecting rubric, PANDA references and AES guidance.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        return labels[AES_KEY] if model == AES_KEY else f"{labels.get(model, model)} / {setup}"

    def line_style(model: Any, setup: str=None) -> dict[str, Any]:
        """Line style.

        Args:
            model (Any): Model used by this operation.
            setup (str): Context setup selecting rubric, PANDA references and AES guidance.

        Returns:
            dict[str, Any]: Named result fields and metadata produced by this operation.
        """
        if model == AES_KEY:
            return {"color": "red", "linestyle": "--", "linewidth": 2, **style.get("aes", {})}
        position = setups.index(setup) if setup in setups else plan["models"].index(model)
        return {"color": palette(position % 10), "linewidth": 1.5 if setup is not None else 1.25,
                **style.get("models", {}).get(model, {}), **style.get("setups", {}).get(setup, {})}

    @contextmanager
    def canvas(title: Any, *, shape: Any=(2, 2), size: Any=(14, 10), sharex: bool=False, sharey: bool=False) -> Iterator[Any]:
        """Canvas.

        Args:
            title (Any): Title used by this operation.
            shape (Any): Shape used by this operation.
            size (Any): Size used by this operation.
            sharex (bool): Sharex used by this operation.
            sharey (bool): Sharey used by this operation.

        Returns:
            Iterator[Any]: The computed result for the supplied inputs.
        """
        fig, axes = plt.subplots(*shape, figsize=size, squeeze=False, sharex=sharex, sharey=sharey)
        try:
            fig.suptitle(title)
            yield fig, axes.flatten()
        finally:
            plt.close(fig)

    def save(fig: Any, stem: Any, *, rect: Any=(0, 0, 1, 0.95)) -> None:
        """Save.

        Args:
            fig (Any): Fig used by this operation.
            stem (Any): Stem used by this operation.
            rect (Any): Rect used by this operation.

        Returns:
            None: This operation returns no value.
        """
        fig.tight_layout(rect=rect)
        for extension in options["formats"]:
            path = output / f"{stem}.{extension}"
            fig.savefig(path, dpi=options["dpi"], bbox_inches="tight", format=extension)
            saved.append(path.name)

    def aes_value(criterion: Any, metric: Any) -> Any:
        """Aes value.

        Args:
            criterion (Any): Criterion used by this operation.
            metric (Any): Metric used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        if not plan.get("include_aes", True):
            return np.nan
        rows = common.loc[common.model_key.eq(AES_KEY) & common.criterion.eq(criterion) & common.metric.eq(metric)]
        return rows["mean"].iloc[0] if len(rows) else np.nan

    def metric_comparison(kind: Any) -> None:
        """Metric comparison.

        Args:
            kind (Any): Kind used by this operation.

        Returns:
            None: This operation returns no value.
        """
        if not n_common:
            return
        models = plan["models"]
        for metric in options["metrics"]:
            selections = [[model] for model in models] if kind == "metric_bars" else [models]
            for selected_models in selections:
                metric_label = "Adjacent accuracy (±1 band)" if metric == "within_one_accuracy" else METRICS[metric]
                title = f"Mean {metric_label} across prompts | common valid n={n_common}"
                comparison_models = selected_models + ([AES_KEY] if plan.get("include_aes", True) else [])
                comparison = common.loc[common.model_key.isin(comparison_models)
                    & common.criterion.isin(CRITERIA) & common.metric.eq(metric)]
                limits = _axis_limits(comparison["mean"].to_numpy(dtype=float), metric)
                for criteria in criterion_groups:
                    with canvas(title, shape=(2, 2) if len(criteria) == 4 else (1, 1),
                                size=(18, 10) if len(criteria) == 4 else (18, 7.5)) as (fig, axes):
                        for ax, criterion in zip(axes, criteria):
                            baseline = aes_value(criterion, metric)
                            if pd.notna(baseline):
                                ax.axhline(baseline, **line_style(AES_KEY))
                            sample = np.arange(len(setups))
                            for model in selected_models:
                                subset = common.loc[common.model_key.eq(model) & common.criterion.eq(criterion) & common.metric.eq(metric)].set_index("setup").reindex(setups)
                                values = pd.to_numeric(subset["mean"], errors="coerce").to_numpy(dtype=float)
                                deviations = pd.to_numeric(subset["std"], errors="coerce").to_numpy(dtype=float)
                                valid = np.isfinite(values)
                                if kind == "metric_bars":
                                    colors = [line_style(model, current_text)["color"] for current_text in setups]
                                    ax.bar(sample, values, color=colors, alpha=0.8)
                                    ax.errorbar(sample, values, yerr=deviations if np.isfinite(deviations).any() else None,
                                                fmt="none", color="black", capsize=3)
                                else:
                                    settings = line_style(model)
                                    ax.scatter(sample[valid], values[valid], s=18, color=settings["color"],
                                               edgecolor="black", linewidth=0.8, zorder=3)
                                    if np.isfinite(deviations).any():
                                        ax.errorbar(sample, values, yerr=deviations, fmt="none", color=settings["color"],
                                                    capsize=4, alpha=0.95, zorder=2)
                                    if valid.sum() >= 2:
                                        ax.plot(sample[valid], values[valid], alpha=0.95, zorder=2,
                                                label=labels.get(model, model), **settings)
                            ax.grid(True, axis="both", alpha=0.5, linewidth=0.8)
                            for separator in np.arange(-0.5, len(setups), 1):
                                ax.axvline(separator, linestyle=":", color="black", alpha=0.25)
                            for index in sample:
                                ax.text(index, -0.01, f"p{index + 1}", transform=ax.get_xaxis_transform(),
                                        ha="center", va="top", fontsize=12, alpha=0.9)
                            ax.set_xticks([])
                            ax.set_title(SHORT_NAMES[criterion], fontweight="bold")
                            ax.set_ylabel(metric_label, fontsize=12)
                            ax.set_xlabel("Prompting configuration", labelpad=20, fontsize=12)
                            ax.axhline(0, linestyle="--", linewidth=1.5, color="black", alpha=0.7, zorder=1)

                        for ax, criterion in zip(axes, criteria):
                            ax.set_ylim(*limits)
                            baseline = aes_value(criterion, metric)
                            if pd.notna(baseline) and limits[0] <= baseline <= limits[1]:
                                ax.text(1.01, baseline, f"{baseline:.3f}", transform=ax.get_yaxis_transform(),
                                        color=line_style(AES_KEY)["color"], fontsize=11, fontweight="bold", va="center")

                        handles = [Patch(facecolor=line_style(model)["color"], edgecolor="black",
                                         label=labels.get(model, model)) for model in selected_models]
                        if plan.get("include_aes", True):
                            values = [aes_value(current_criterion, metric) for current_criterion in CRITERIA]
                            numbers = [f"{current_value:.3f}" if pd.notna(current_value) else "undefined" for current_value in values]
                            aes_label = f"AES ({METRICS[metric]}; TR / CC / LR / GRA)\n" + " / ".join(numbers)
                            handles.append(Line2D([0], [0], label=aes_label, **line_style(AES_KEY)))
                        anchor = 0.4 if len(setups) == 6 else 0.45
                        fig.legend(handles=handles, title="Models", loc="upper right", bbox_to_anchor=(anchor, 1),
                                   ncol=min(len(handles), 3), fontsize=12)
                        prompt_handles = [Line2D([0], [0], marker="o", linestyle="None", color="black",
                                                 label=f"p{current_index + 1}: {setup}") for current_index, setup in enumerate(setups)]
                        fig.legend(handles=prompt_handles, title="Prompting configurations", loc="upper left",
                                   bbox_to_anchor=(anchor, 1), ncol=3 if len(setups) == 6 else 2, fontsize=12)
                        fig.suptitle(title, fontsize=16, y=0.9 if len(criteria) == 4 else 0.85)
                        suffix = "_" + selected_models[0] if kind == "metric_bars" else ""
                        stems = {"qwk": "mean_qwk_across_runs", "mae": "mean_mae_across_runs", "rmse": "mean_rmse_across_runs",
                                 "pearson": "mean_pearson_corr_across_runs", "spearman": "mean_spearman_corr_across_runs",
                                 "within_one_accuracy": "adjacent_accuracy_across_runs"}
                        stem = stems.get(metric, metric + "_across_runs") if kind == "metric_comparison" else f"{kind}_{metric}{suffix}"
                        if len(criteria) == 1:
                            stem += "_" + SHORT_NAMES[criteria[0]]
                        save(fig, stem, rect=(0, 0, 1, 0.90 if len(criteria) == 4 else 0.85))

    def heatmaps() -> None:
        """Heatmaps.

        Returns:
            None: This operation returns no value.
        """
        if not n_common:
            return
        for metric in options["metrics"]:
            data = common.loc[common.metric.eq(metric) & common.criterion.isin(CRITERIA)]
            matrix = np.array([[data.loc[data.model_key.eq(current_match) & data.setup.eq(current_text) & data.criterion.eq(current_criterion), "mean"].iloc[0]
                                for current_criterion in CRITERIA] for current_match, current_text in series], dtype=float)
            with canvas(f"{METRICS[metric]} | common valid n={n_common}", shape=(1, 1), size=(10, max(4, len(series) * 0.55))) as (fig, axes):
                ax = axes[0]
                im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="viridis")
                ax.set_xticks(range(4), [SHORT_NAMES[current_criterion] for current_criterion in CRITERIA])
                ax.set_yticks(range(len(series)), [name(*key) for key in series])
                for (current_index, column_index), value in np.ndenumerate(matrix):
                    ax.text(column_index, current_index, f"{value:.3f}" if np.isfinite(value) else "undefined", ha="center", va="center", fontsize=8)
                fig.colorbar(im, ax=ax)
                save(fig, f"metric_heatmap_{metric}")

    def curves(kind: Any) -> None:
        """Curves.

        Args:
            kind (Any): Kind used by this operation.

        Returns:
            None: This operation returns no value.
        """
        table = tables["calibration"] if kind == "calibration" else tables["score_response"]
        column = {"calibration": "mean_reference", "score_response": "mean_prediction", "bias": "bias"}[kind]
        for model in plan["models"] or [AES_KEY]:
            title = {"score_response": "Calibration curves", "calibration": "Calibration by predicted score", "bias": "Mean bias"}[kind]
            original_calibration = kind == "score_response"
            if original_calibration:
                points = table.loc[table.model_key.eq(model) & table.setup.isin(setups)
                    & table.criterion.isin(CRITERIA)].groupby(["setup", "criterion", "band"])[column].mean()
                limits = _axis_limits(points.to_numpy(dtype=float), "calibration")
            groups = criterion_groups if original_calibration else [CRITERIA]
            for criteria in groups:
                with canvas(f"{labels.get(model, model)} | {title}", shape=(2, 2) if len(criteria) == 4 else (1, 1),
                            size=(14, 10) if len(criteria) == 4 else (14, 7.5),
                            sharex=original_calibration, sharey=original_calibration) as (fig, axes):
                    for ax, criterion in zip(axes, criteria):
                        bands = (tables["reference_distribution"].loc[
                            lambda rows: rows.criterion.eq(criterion) & rows["count"].gt(0), "band"].to_numpy(dtype=int)
                            if original_calibration else np.arange(10))
                        curve_series = [(model, current_text) for current_text in setups]
                        if not original_calibration and plan.get("include_aes", True):
                            curve_series.insert(0, (AES_KEY, "aes"))
                        for current_match, setup in curve_series:
                            subset = table.loc[table.model_key.eq(current_match) & table.setup.eq(setup) & table.criterion.eq(criterion)]
                            grouped = subset.groupby("band")[column].agg(["mean", "std"]).reindex(bands)
                            deviations = grouped["std"].to_numpy(dtype=float)
                            ax.errorbar(bands, grouped["mean"], yerr=deviations if np.isfinite(deviations).any() else None, marker="o", capsize=3,
                                        label=setup if original_calibration else name(current_match, setup), **line_style(current_match, setup))
                        if kind == "bias":
                            ax.axhline(0, color="black", linewidth=1)
                        else:
                            ax.plot([0, 9], [0, 9], color="black", linestyle="--" if original_calibration else ":")
                        ax.set_xticks(range(10))
                        ax.set_title(SHORT_NAMES[criterion])
                        ax.set_xlabel("Predicted band" if kind == "calibration" else "Source band")
                        ax.set_ylabel({"calibration": "Mean source score", "score_response": "Mean prediction", "bias": "Mean prediction - source"}[kind])
                        if original_calibration:
                            ax.set_yticks(range(10))
                            ax.grid(True, alpha=1, linewidth=0.8)
                            ax.tick_params(labelbottom=True, labelleft=True)
                            ax.set_xlabel("")
                            ax.set_ylabel("")
                    if original_calibration:
                        for ax in axes:
                            ax.set_ylim(*limits)
                        handles, curve_labels = axes[0].get_legend_handles_labels()
                        fig.legend(handles, curve_labels, loc="lower center", bbox_to_anchor=(0.5, 0.96),
                                   ncol=min(len(setups), 3), title="Prompt configurations")
                        fig.suptitle(f"{labels.get(model, model)} | Calibration curves", fontsize=14, y=0.95)
                        fig.supxlabel("Source band")
                        fig.supylabel("Mean prediction")
                    else:
                        axes[0].legend(fontsize=7)
                    stem = "calibration" if kind == "score_response" else "calibration_by_prediction" if kind == "calibration" else "bias"
                    panel = "grid" if len(criteria) == 4 else SHORT_NAMES[criteria[0]]
                    save(fig, f"{model}_{stem}_{panel}", rect=(0, 0, 0.9, 0.95) if original_calibration else (0, 0, 1, 0.95))

    def histograms(kind: Any) -> None:
        """Histograms.

        Args:
            kind (Any): Kind used by this operation.

        Returns:
            None: This operation returns no value.
        """
        table = tables["error_distribution"]
        for model in plan["models"] or [AES_KEY]:
            if kind == "error_histogram_std" and plan["repeats"] < 2:
                continue
            if kind == "error_ridge":
                for criterion in CRITERIA:
                    with canvas(f"{labels.get(model, model)} | {SHORT_NAMES[criterion]} | error frequencies", shape=(1, 1), size=(11, 7)) as (fig, axes):
                        ax = axes[0]
                        for current_index, setup in enumerate(setups):
                            data = table.loc[table.model_key.eq(model) & table.setup.eq(setup) & table.criterion.eq(criterion)]
                            values = data.groupby("error")["fraction"].mean().reindex(range(-9, 10))
                            offset = current_index * 1.1
                            ax.step(values.index, values + offset, where="mid", **line_style(model, setup))
                            ax.fill_between(values.index, offset, values + offset, step="mid", alpha=0.15,
                                            color=line_style(model, setup)["color"])
                        ax.set_yticks(np.arange(len(setups)) * 1.1, setups)
                        ax.axvline(0, color="black", linestyle=":")
                        ax.set_xlabel("Prediction - source score (integer bins, no smoothing)")
                        save(fig, f"error_ridge_{model}_{SHORT_NAMES[criterion]}")
                continue
            with canvas(f"{labels.get(model, model)} | {kind} | fixed integer bins; own valid observations") as (fig, axes):
                for ax, criterion in zip(axes, CRITERIA):
                    for current_match, setup in [(AES_KEY, "aes"), *[(model, current_text) for current_text in setups]]:
                        if kind == "error_histogram_std" and current_match == AES_KEY:
                            continue
                        data = table.loc[table.model_key.eq(current_match) & table.setup.eq(setup) & table.criterion.eq(criterion)]
                        grouped = data.groupby("error")["fraction"]
                        values = grouped.std(ddof=1) if kind == "error_histogram_std" else grouped.mean()
                        ax.step(values.index, values, where="mid", label=name(current_match, setup), **line_style(current_match, setup))
                    ax.axvline(0, color="black", linestyle=":")
                    ax.set_xticks(range(-9, 10, 3))
                    ax.set_title(SHORT_NAMES[criterion])
                    ax.set_xlabel("Prediction - source score")
                    ax.set_ylabel("Across-run STD of fractions" if kind == "error_histogram_std" else "Fraction")
                axes[0].legend(fontsize=7)
                save(fig, f"{kind}_{model}")

    def confusion_matrices() -> None:
        """Confusion matrices.

        Returns:
            None: This operation returns no value.
        """
        table = tables["confusion"]
        for model, setup in series:
            for column in ("count", "row_fraction"):
                with canvas(f"{name(model, setup)} | confusion {column} | own valid observations") as (fig, axes):
                    for ax, criterion in zip(axes, CRITERIA):
                        data = table.loc[table.model_key.eq(model) & table.setup.eq(setup) & table.criterion.eq(criterion)]
                        matrix = data.groupby(["true_band", "predicted_band"])[column].mean().unstack().reindex(index=range(10), columns=range(10))
                        im = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), cmap="Blues", vmin=0,
                                       vmax=1 if column == "row_fraction" else None)
                        for (current_index, column_index), value in np.ndenumerate(matrix.to_numpy(dtype=float)):
                            if np.isfinite(value) and value > 0:
                                ax.text(column_index, current_index, f"{value:.0%}" if column == "row_fraction" else f"{value:g}", ha="center", va="center", fontsize=6)
                        ax.set_xticks(range(10))
                        ax.set_yticks(range(10))
                        ax.set_xlabel("Prediction")
                        ax.set_ylabel("Source band")
                        ax.set_title(SHORT_NAMES[criterion])
                        fig.colorbar(im, ax=ax, fraction=0.04)
                    save(fig, f"confusion_{model}_{setup}_{column}")

    def score_distribution() -> None:
        """Score distribution.

        Returns:
            None: This operation returns no value.
        """
        if not n_common:
            return
        frame = tables["predictions"]
        valid = frame.loc[frame.status.eq("ok") & frame.essay_id.isin(tables["common_ids"].essay_id)]
        for model in plan["models"] or [AES_KEY]:
            with canvas(f"{labels.get(model, model)} | score distributions | common valid n={n_common}") as (fig, axes):
                for ax, criterion in zip(axes, CRITERIA):
                    actual = valid.drop_duplicates("essay_id")["true_" + criterion]
                    counts = actual.value_counts(normalize=True).reindex(range(10), fill_value=0)
                    ax.bar(range(10), counts, color="lightgray", label="Source labels")
                    for current_match, setup in [(AES_KEY, "aes"), *[(model, current_text) for current_text in setups]]:
                        data = valid.loc[valid.model_key.eq(current_match) & valid.setup.eq(setup), criterion]
                        counts = data.value_counts(normalize=True).reindex(range(10), fill_value=0)
                        if len(data):
                            ax.plot(range(10), counts, marker="o", label=name(current_match, setup), **line_style(current_match, setup))
                    ax.set_title(SHORT_NAMES[criterion])
                    ax.set_xticks(range(10))
                    ax.set_ylabel("Fraction")
                axes[0].legend(fontsize=7)
                save(fig, f"score_distribution_{model}")

    def coverage() -> None:
        """Coverage.

        Returns:
            None: This operation returns no value.
        """
        table = tables["coverage"].groupby(["model_key", "setup"], sort=False)[["planned", "processed", "valid"]].sum()
        with canvas("Coverage over all planned requests", shape=(1, 1), size=(12, 6)) as (fig, axes):
            ax = axes[0]
            planned = table.planned.to_numpy(dtype=float)
            values = [table.valid.to_numpy() / planned, (table.processed - table.valid).to_numpy() / planned,
                      (table.planned - table.processed).to_numpy() / planned]
            bottom = np.zeros(len(table))
            for label, value, color in zip(("Valid", "Failed", "Not run"), values, ("#4c956c", "#c44536", "#bdbdbd")):
                ax.bar(range(len(table)), value, bottom=bottom, label=label, color=color)
                bottom += value
            ax.set_xticks(range(len(table)), [name(current_match, current_text) for current_match, current_text in table.index], rotation=30, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Fraction of planned requests")
            ax.legend()
            save(fig, "coverage")

    def prompt_comparison() -> None:
        """Prompt comparison.

        Returns:
            None: This operation returns no value.
        """
        if "base" not in setups or not n_common:
            return
        for metric in options["metrics"]:
            with canvas(f"Change in {METRICS[metric]} relative to base | common valid n={n_common}") as (fig, axes):
                for ax, criterion in zip(axes, CRITERIA):
                    for model in plan["models"]:
                        data = common.loc[common.model_key.eq(model) & common.criterion.eq(criterion) & common.metric.eq(metric)].set_index("setup").reindex(setups)
                        baseline = data.loc["base", "mean"]
                        ax.plot(range(len(setups)), data["mean"] - baseline, marker="o", label=labels.get(model, model), **line_style(model))
                    ax.axhline(0, color="black", linestyle=":")
                    ax.set_xticks(range(len(setups)), setups, rotation=25, ha="right")
                    ax.set_title(SHORT_NAMES[criterion])
                    ax.set_ylabel(f"Delta {METRICS[metric]} (setup - base)")
                axes[0].legend(fontsize=8)
                save(fig, f"comparison_across_prompts_{metric}")

    def repeatability() -> None:
        """Repeatability.

        Returns:
            None: This operation returns no value.
        """
        table = tables["repeatability"]
        measured = table.loc[table.valid_repeats.ge(2)]
        if measured.empty:
            return
        with canvas("Repeatability | essays with at least two valid repetitions") as (fig, axes):
            for ax, criterion in zip(axes, CRITERIA):
                data = measured.loc[measured.criterion.eq(criterion)].groupby(["model_key", "setup"], sort=False)["all_equal"].mean()
                ax.bar(range(len(data)), data.astype(float))
                ax.set_xticks(range(len(data)), [name(current_match, current_text) for current_match, current_text in data.index], rotation=30, ha="right")
                ax.set_ylim(0, 1.05)
                ax.set_title(SHORT_NAMES[criterion])
                ax.set_ylabel("Fraction of identical criterion predictions")
            save(fig, "repeatability")

    def latency() -> None:
        """Latency.

        Returns:
            None: This operation returns no value.
        """
        table = tables["predictions"]
        if "judge_seconds" not in table:
            return
        selected = table.loc[table.model_key.ne(AES_KEY) & table.status.eq("ok")].dropna(subset=["judge_seconds"])
        groups = list(selected.groupby(["model_key", "setup"], sort=False))
        if not groups:
            return
        with canvas("Local model generation time | valid requests, prepared context", shape=(1, 1), size=(12, 6)) as (fig, axes):
            ax = axes[0]
            ax.boxplot([current_group.judge_seconds.to_numpy(dtype=float) for _, current_group in groups], tick_labels=[name(*key) for key, _ in groups])
            ax.tick_params(axis="x", rotation=30)
            ax.set_ylabel("Judge wall time, seconds")
            save(fig, "latency")

    handlers = {
        "metric_comparison": lambda: metric_comparison("metric_comparison"),
        "metric_bars": lambda: metric_comparison("metric_bars"), "metric_heatmap": heatmaps,
        "calibration": lambda: curves("score_response"), "calibration_by_prediction": lambda: curves("calibration"),
        "bias": lambda: curves("bias"), "error_histogram": lambda: histograms("error_histogram"),
        "error_histogram_std": lambda: histograms("error_histogram_std"), "error_ridge": lambda: histograms("error_ridge"),
        "confusion_matrix": confusion_matrices, "score_distribution": score_distribution,
        "coverage": coverage, "prompt_comparison": prompt_comparison, "repeatability": repeatability, "latency": latency,
    }
    with plt.rc_context(style.get("rc", {})):
        for plot_type in options["plots"]:
            before = len(saved)
            handlers[plot_type]()
            if len(saved) == before:
                skipped.append({"type": plot_type, "reason": "required common-valid observations, baseline, timings or multiple valid repeats unavailable"})
    return {"files": saved, "skipped": skipped}
