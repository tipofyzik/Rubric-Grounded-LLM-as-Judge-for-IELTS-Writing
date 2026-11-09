"""Inspect raw essay sources and plot score distributions and example lengths."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math
from pathlib import Path

from ielts_judge.Paths import PATHS

class RawDataAnalyzer:
    """Analyzes the raw dataset to give a hint where the preprocess should start first.

    Attributes:
        __dataset (pd.DataFrame): The raw dataset to be analyzed.
        __dataset_name (str): The name of the provided dataset.
    """

    def __init__(self, dataset: pd.DataFrame, dataset_name: str) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            dataset (pd.DataFrame): Dataset used by this operation.
            dataset_name (str): Dataset name used by this operation.

        Returns:
            None: This operation returns no value.
        """
        self.__dataset = dataset
        self.__dataset_name = dataset_name

    def __get_dataset_shape(self) -> tuple[int, int]:
        """Retrieves the shape of the dataset.

        Returns:
            tuple[int, int]: The computed result for the supplied inputs.
        """
        return self.__dataset.shape

    def print_dataset_shape(self) -> None:
        """Prints the shape of the dataset.

        Returns:
            None: This operation returns no value.
        """
        print(f"The shape of the \'{self.__dataset_name}\' is {self.__get_dataset_shape()}")

    def __check_null_values(self) -> bool:
        """Sums up all the null values found in the table and calculates
        the boolean value that reflects the presence of null values.

        Returns:
            bool: The computed result for the supplied inputs.
        """
        self.__null_sum = self.__dataset.isnull().sum().sum()
        self.__have_null = bool(self.__null_sum)
        return self.__have_null

    def print_rows_with_null(self) -> None:
        """Prints the rows that contain null values.

        Returns:
            None: This operation returns no value.
        """
        null_rows = self.__dataset[self.__dataset.isnull().any(axis=1)]

        if null_rows.empty:
            print(f"The '{self.__dataset_name}' does not have any null values.")
            return

        for current_index, row in null_rows.iterrows():
            null_count = row.isnull().sum()
            print(f"Row index: {current_index} | Null values in this row: {null_count}")

    def print_column_names(self) -> None:
        """Prints the list that contains sheet's column names.

        Returns:
            None: This operation returns no value.
        """
        print(self.__dataset.columns)

    def plot_score_distributions(self, criteria: list[str], save_dir: str) -> None:
        """Plots and saves the distribution of essay counts by score for each criterion.

        Args:
            criteria (list[str]): Criteria used by this operation.
            save_dir (str): Save dir used by this operation.

        Returns:
            None: This operation returns no value.
        """
        valid_data = []
        labels = []

        for criterion in criteria:
            if criterion not in self.__dataset.columns:
                print(f"Column '{criterion}' not found in dataset.")
                continue

            data = self.__dataset[criterion].dropna()
            valid_data.append(data)
            labels.append(criterion.replace("_", " ").title())

        # размеры сетки
        current_count = len(valid_data)
        current_columns = 2
        rows = math.ceil(current_count / current_columns)

        fig, axes = plt.subplots(rows, current_columns, figsize=(12, 4 * rows))
        axes = axes.flatten()

        # общий диапазон для честного сравнения
        global_min = min(current_record.min() for current_record in valid_data)
        global_max = max(current_record.max() for current_record in valid_data)

        bin_edges = np.arange(global_min, global_max + 0.5, 0.5)

        # рисуем
        for ax, data, label in zip(axes, valid_data, labels):
            ax.hist(
                data,
                bins=bin_edges.tolist(),
                edgecolor="black",
                alpha=0.85
            )

            ax.set_title(label, fontsize=12, fontweight="bold")

            ax.set_xlabel("Score")
            ax.set_ylabel("Number of essays")

            ax.set_xticks(range(0, 10, 1))
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        # убираем пустые subplot'ы
        for current_index in range(len(valid_data), len(axes)):
            fig.delaxes(axes[current_index])

        fig.suptitle("Score distributions across criteria", fontsize=16, fontweight="bold")

        plt.tight_layout(rect=(0, 0, 1, 0.96))

        file_path = Path(save_dir) / PATHS["files"]["score_distribution_plot"]
        plt.savefig(file_path, dpi=250, bbox_inches="tight", format="webp")
        plt.close()
