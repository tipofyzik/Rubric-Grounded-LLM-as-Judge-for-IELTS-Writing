import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math

class RawDataAnalyzer:
    """
    Analyzes the raw dataset to give a hint where the preprocess should start first.

    Attributes: 
        __dataset (pd.DataFrame): The raw dataset to be analyzed.
        __dataset_name (str): The name of the provided dataset.
    """

    def __init__(self, dataset: pd.DataFrame, dataset_name: str) -> None:
        """
        Initializes the RawDataAnalyzer with the provided dataset and its name.

        Args:
            dataset (pd.DataFrame): The raw dataset to be analyzed.
            dataset_name (str): The name of the provided dataset.
        """
        self.__dataset = dataset
        self.__dataset_name = dataset_name

    def __get_dataset_shape(self) -> tuple[int, int]:
        """
        Retrieves the shape of the dataset. 

        Returns:
            (tuple[int, int]): The shape of the dataset.
        """
        return self.__dataset.shape

    def print_dataset_shape(self) -> None:
        """
        Prints the shape of the dataset.
        """
        print(f"The shape of the \'{self.__dataset_name}\' is {self.__get_dataset_shape()}")

    def __check_null_values(self) -> bool:
        """
        Sums up all the null values found in the table and calculates 
        the boolean value that reflects the presence of null values.

        Returns:
            (bool): The value that represents whether there are null values in the dataset.
        """
        self.__null_sum = self.__dataset.isnull().sum().sum()
        self.__have_null = bool(self.__null_sum)
        return self.__have_null
    
    def print_rows_with_null(self) -> None:
        """
        Prints the rows that contain null values.
        """
        null_rows = self.__dataset[self.__dataset.isnull().any(axis=1)]

        if null_rows.empty:
            print(f"The '{self.__dataset_name}' does not have any null values.")
            return

        for idx, row in null_rows.iterrows():
            null_count = row.isnull().sum()
            print(f"Row index: {idx} | Null values in this row: {null_count}")

    def print_column_names(self) -> None:
        """
        Prints the list that contains sheet's column names.
        """
        print(self.__dataset.columns)

    def plot_score_distributions(self, criteria: list[str], save_dir: str) -> None:
        """
        Plots and saves the distribution of essay counts by score for each criterion.

        Args:
            criteria (list[str]): List of column names representing scoring criteria.
            save_dir (str): Directory where plots will be saved.
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
        n = len(valid_data)
        cols = 2
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        # общий диапазон для честного сравнения
        global_min = min(d.min() for d in valid_data)
        global_max = max(d.max() for d in valid_data)

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
        for i in range(len(valid_data), len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle("Score distributions across criteria", fontsize=16, fontweight="bold")

        plt.tight_layout(rect=(0, 0, 1, 0.96))

        file_path = f"{save_dir}/score_distribution_grid.webp"
        plt.savefig(file_path, dpi=250, bbox_inches="tight", format="webp")
        plt.close()