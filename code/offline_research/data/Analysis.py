"""Audit and plots for the common Parquet schema."""
from typing import Any

from collections import Counter
import math
from pathlib import Path

from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from offline_research.data.Profiles import count_score_profiles, score_profile_ids


class PreparedDataAnalyzer:
    """Manage prepared data analyzer through the operations defined by this component."""
    def __init__(self, frame: Any) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            frame (Any): Frame used by this operation.

        Returns:
            None: This operation returns no value.
        """
        self.frame = frame
        self.accepted = frame.loc[frame["status"].eq("accepted")]

    def summary(self) -> dict[str, Any]:
        """Summary.

        Returns:
            dict[str, Any]: Named result fields and metadata produced by this operation.
        """
        def counts(series: Any) -> dict[str, Any]:
            """Counts.

            Args:
                series (Any): Series used by this operation.

            Returns:
                dict[str, Any]: Named result fields and metadata produced by this operation.
            """
            return {str(current_key): int(current_value) for current_key, current_value in series.value_counts().items()}
        cross_hashes = self.frame.groupby("essay_hash")["dataset_source"].nunique()
        cross = self.frame.loc[self.frame["essay_hash"].isin(cross_hashes.index[cross_hashes.gt(1)])]
        cross_conflicts = cross.loc[cross["issues"].map(lambda values: "duplicate_essay_conflict" in values)]
        return {
            "rows": len(self.frame), "status_counts": counts(self.frame["status"]),
            "source_counts": counts(self.frame["dataset_source"]),
            "accepted_source_counts": counts(self.accepted["dataset_source"]),
            "source_split_counts": counts(self.frame["source_split"]),
            "accepted_split_counts": counts(self.accepted["split"]),
            "issue_counts": dict(sorted(Counter(issue for items in self.frame["issues"] for issue in items).items())),
            "warning_counts": dict(sorted(Counter(issue for items in self.frame["warnings"] for issue in items).items())),
            "cross_source_duplicates": {"essay_groups": int(cross["essay_hash"].nunique()),
                                        "rows": len(cross), "conflicting_groups": int(cross_conflicts["essay_hash"].nunique()),
                                        "rows_with_conflicting_scores": len(cross_conflicts)},
            "sources": {name: PreparedDataAnalyzer(group).source_summary()
                        for name, group in self.frame.groupby("dataset_source", sort=True)},
        }

    def source_summary(self) -> dict[str, Any]:
        """Source summary.

        Returns:
            dict[str, Any]: Named result fields and metadata produced by this operation.
        """
        scores = {}
        for name in PUBLIC_CRITERION_IDS:
            values = self.accepted[name].dropna()
            scores[name] = {"count": len(values), "histogram": {str(current_index): int(values.eq(current_index).sum()) for current_index in range(10)}}
        return {"rows": len(self.frame), "accepted": len(self.accepted),
                "issue_counts": dict(sorted(Counter(issue for items in self.frame["issues"] for issue in items).items())),
                "status_counts": {str(current_key): int(current_value) for current_key, current_value in self.frame["status"].value_counts().items()},
                "scores": scores, "overall_labels": {str(current_key): int(current_value) for current_key, current_value in self.accepted["overall_label"].value_counts().items()}}

    def profiles(self, *, with_ids: bool=False) -> Any:
        """Profiles.

        Args:
            with_ids (bool): With ids used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        analyze = score_profile_ids if with_ids else count_score_profiles
        report = analyze(self.accepted, PUBLIC_CRITERION_IDS)
        report["by_source"] = {
            source: analyze(group, PUBLIC_CRITERION_IDS, 4 if source == "writing9" else 0, 9)
            for source, group in self.accepted.groupby("dataset_source", sort=True)}
        report["by_split"] = {
            split: analyze(self.accepted.loc[self.accepted["split"].eq(split)], PUBLIC_CRITERION_IDS)
            for split in ("train", "test")}
        return report

    def plot_score_distributions(self, output: Any) -> Any:
        """Plot score distributions.

        Args:
            output (Any): Output used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        output = Path(output)
        paths = []
        groups = [("combined", self.accepted),
                  *((source, self.accepted.loc[self.accepted['dataset_source'].eq(source)])
                    for source in sorted(self.frame['dataset_source'].unique()))]
        for source, frame in groups:
            for criterion in (*PUBLIC_CRITERION_IDS, "overall"):
                histogram = dict.fromkeys((str(current_index) for current_index in range(10)), 0)
                if criterion == "overall":
                    labels = []
                    for value in frame["overall_label"].dropna():
                        if value == "<4":
                            labels.append(value)
                        else:
                            labels.append(str(math.trunc(float(value))))
                    histogram.update(Counter(labels))
                    if "<4" in histogram:
                        histogram = {"<4": histogram.pop("<4"), **histogram}
                    histogram = {label: count for label, count in histogram.items() if count} or {"<4": 0}
                else:
                    histogram.update({str(int(current_key)): int(current_value) for current_key, current_value in frame[criterion].dropna().value_counts().items()})
                figure = Figure(figsize=(8, 5), dpi=150, layout="constrained")
                FigureCanvasAgg(figure)
                axis = figure.subplots()
                axis.bar(range(len(histogram)), list(histogram.values()), zorder=3)
                axis.set_xticks(range(len(histogram)), list(histogram))
                axis.set_xlim(-0.5, len(histogram) - 0.5)
                axis.set_title(f"{source}: {criterion.replace('_', ' ').title()} (n={sum(histogram.values()):,})")
                axis.set_xlabel("Source overall label (numeric values truncated)" if criterion == "overall" else "Integer criterion score")
                axis.set_ylabel("Number of essays")
                axis.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
                path = output / source / f"score_distribution_{criterion}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(path, dpi=150)
                figure.clear()
                paths.append(path)
        return paths
