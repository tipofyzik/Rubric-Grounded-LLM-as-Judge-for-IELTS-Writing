import os
import json
import numpy as np
import pandas as pd

from collections import defaultdict, Counter



class LLMAggregator:
    """
    Aggregates multiple JSON files containing LLM evaluation outputs,
    handles missing values, computes statistics, and exports results.
    """
    def __init__(self, evaluation_folder_path: str):
        """
        Initialize the aggregator.

        Args:
            evaluation_folder_path (str): Path to directory with JSON result files.
            models (defaultdict): Dictionary to store loaded data grouped by model name.
        """
        self.folder_path = evaluation_folder_path
        self.models = defaultdict(list)

    # ---------- LOAD ----------
    def __load_files(self):
        """
        Load JSON files from the folder and group them by model name.
        """
        for file in os.listdir(self.folder_path):
            if file.endswith(".json"):
                model_name = "_".join(file.split("_")[:-1])
                full_path = os.path.join(self.folder_path, file)

                with open(full_path, "r", encoding="utf-8") as file:
                    file_data = json.load(file)
                self.models[model_name].append(file_data)

    # ---------- EXTRACT ----------
    def __extract_scores(self, model_output: dict) -> dict:
        """
        Return scores dict from model output.

        Supports two formats:
        - {"answer": {...}} → returns model_output["answer"]
        - {...} → returns scores directly

        Non-dictionary input → empty dictionary.
        """

        if isinstance(model_output, dict):
            # Contains "answer" field with dict value → return that
            if "answer" in model_output and isinstance(model_output["answer"], dict):
                return model_output["answer"]

            # Contains "answer" field but answer is null → return empty dict
            # To cover the case when {"answer": null} is used to indicate missing data
            elif "answer" in model_output and model_output["answer"] is None: 
                return {}

            # No "answer" field → return entire output
            return model_output
        return {}
        
    # ---------- MODE ----------
    def __compute_mode_for_file(self, file_data: list) -> dict:
        """
        Compute the mode value for each field in a single file.

        Args:
            file_data (list): List of evaluation entries.

        Returns:
            dict: Mapping of field names to their mode values.
        """
        criteria_scores = defaultdict(list)

        for evaluation in file_data:
            scores = self.__extract_scores(evaluation)
            for criteria, score in scores.items():
                if score is not None:
                    criteria_scores[criteria].append(score)

        mode_dict = {}
        for criteria, values in criteria_scores.items():
            if values:
                mode_dict[criteria] = Counter(values).most_common(1)[0][0]
            else:
                mode_dict[criteria] = 0.0

        return mode_dict

    # ---------- CLEAN ----------
    def __clean_nulls(self) -> None:
        """
        Entry point for null-value cleaning across all models and files.

        Iterates over all stored models and delegates cleaning of each
        file to `_clean_model_files`.

        Missing values are replaced using per-file mode statistics.
        """
        for model, files in self.models.items():
            self.__clean_model_files(model, files)

    def __clean_model_files(self, model: str, files: list) -> None:
        """
        Cleans all files for a given model.

        For each file:
        - Computes mode statistics based on file content
        - Applies cleaning logic to each item using `_clean_file`

        Args:
            model: Model identifier key in `self.models`.
            files: List of file datasets associated with the model.
        """
        for file_index, file_data in enumerate(files):
            mode_dict = self.__compute_mode_for_file(file_data)
            self.__clean_file(model, file_index, file_data, mode_dict)

    def __clean_file(self, model: str, file_idx: int, 
                    file_data: list, mode_dict: dict) -> None:
        """
        Cleans a single file dataset in-place.

        Iterates through all items in the file and replaces
        missing values using computed mode statistics.

        Args:
            model: Model identifier.
            file_idx: Index of the file within the model.
            file_data: List of data items in the file.
            mode_dict: Precomputed mode values for filling missing data.
        """
        for i, item in enumerate(file_data):
            cleaned = self.__clean_item(item, mode_dict)

            if cleaned is not None:
                self.models[model][file_idx][i] = cleaned

    def __clean_item(self, item, mode_dict: dict):
        """
        Cleans a single data item.

        Rules:
        - If item is None → replace with mode dictionary copy
        - If item is a dict containing "answer" → clean only answer field
        - Otherwise → fill missing values directly in item dict

        Args:
            item: Data item (dict or None or other structure).
            mode_dict: Mode-based fallback values.

        Returns:
            Cleaned item or replacement dict if item was None.
        """
        if item is None:
            return mode_dict.copy()

        if not isinstance(item, dict):
            return self.__fill_missing_values(item, mode_dict)

        if "answer" in item:
            self.__clean_answer_field(item["answer"], mode_dict)
            return item

        return self.__fill_missing_values(item, mode_dict)

    def __clean_answer_field(self, answer: dict, mode_dict: dict) -> None:
        """
        Cleans the "answer" sub-dictionary in-place.

        Replaces None values inside the answer dict with
        corresponding mode values.

        Args:
            answer: Dictionary containing answer fields.
            mode_dict: Mode-based fallback values.
        """
        if answer is None:
            return mode_dict.copy()

        for k, v in answer.items():
            if v is None:
                answer[k] = mode_dict[k]

    def __fill_missing_values(self, item: dict, mode_dict: dict) -> dict:
        """
        Fills missing (None) values in a dictionary in-place.

        Iterates over all keys and replaces None values using
        corresponding entries from mode_dict.

        Args:
            item: Dictionary to clean.
            mode_dict: Mode-based fallback values.

        Returns:
            The same dictionary instance with missing values filled.
        """
        for k, v in item.items():
            if v is None:
                item[k] = mode_dict[k]
        return item
    
    # ---------- STATS ----------
    def __compute_stats(self) -> pd.DataFrame:
        """
        Compute mean and standard deviation across files for each model.

        Returns:
            pd.DataFrame: Aggregated statistics.
        """
        all_results = []

        for model, files in self.models.items():

            lengths = [len(f) for f in files]
            if len(set(lengths)) != 1:
                raise ValueError(f"Inconsistent lengths in model {model}")

            num_points = lengths[0]

            first_item = self.__extract_scores(files[0][0])
            if first_item is None:
                raise ValueError("First item contains no valid scores")
            keys = list(first_item.keys())

            data = np.array([
                [
                    [self.__extract_scores(item)[k] for k in keys]
                    for item in file_data
                ]
                for file_data in files
            ])

            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            for i in range(num_points):
                row = {
                    "model": model,
                    "index": i
                }

                for j, k in enumerate(keys):
                    row[f"{k}_mean"] = mean[i, j]
                    row[f"{k}_std"] = std[i, j]

                all_results.append(row)

        return pd.DataFrame(all_results)

    # ---------- SAVE ----------
    def __save_to_csv(self, output_path: str) -> None:
        """
        Save computed statistics to a CSV file.

        Args:
            output_path (str): Output file path or directory.
        """
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "results.csv")

        df = self.__compute_stats()
        df.to_csv(output_path, index=False)

    def __save_all_to_csv(self, output_path: str) -> None:
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "results.csv")

        rows = []

        for model, files in self.models.items():

            for file_idx, file_data in enumerate(files):

                for sample_idx, item in enumerate(file_data):

                    scores = self.__extract_scores(item)

                    if not scores:
                        continue

                    row = {
                        "model": model,
                        "file_id": file_idx,
                        "sample_id": sample_idx
                    }

                    for k, v in scores.items():
                        row[k] = v

                    rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    # ---------- RUN ----------
    def run(self, output_path: str = "./results.csv") -> None:
        """
        Execute full aggregation pipeline.

        Args:
            output_path (str): Output file path.
        """
        self.__load_files()
        self.__clean_nulls()
        self.__save_all_to_csv(output_path)
