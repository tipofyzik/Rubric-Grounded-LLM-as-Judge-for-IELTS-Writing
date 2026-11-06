from typing import Any
import os
import json
import numpy as np
import pandas as pd

from ielts_judge.Paths import PATHS
from collections import defaultdict, Counter
from offline_research.data.Runtime import legacy_prediction



class LLMAggregator:
    """Aggregates multiple JSON files containing LLM evaluation outputs,
    handles missing values, computes statistics, and exports results.
    """
    def __init__(self, evaluation_folder_path: str) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            evaluation_folder_path (str): Evaluation folder path used by this operation.

        Returns:
            None: This operation returns no value.
        """
        self.folder_path = evaluation_folder_path
        self.models = defaultdict(list)

    # ---------- LOAD ----------
    def __load_files(self) -> None:
        """Load JSON files from the folder and group them by model name.

        Returns:
            None: This operation returns no value.
        """
        for file in sorted(os.listdir(self.folder_path)):
            if file.endswith(".json"):
                model_name = "_".join(file.split("_")[:-1])
                full_path = os.path.join(self.folder_path, file)

                with open(full_path, "r", encoding="utf-8") as file:
                    file_data = json.load(file)
                self.models[model_name].append(file_data)

    # ---------- EXTRACT ----------
    def __extract_scores(self, model_output: dict) -> dict:
        """Return scores dict from model output.

        Supports two formats:
        - {"answer": {...}} → returns model_output["answer"]
        - {...} → returns scores directly

        Non-dictionary input → empty dictionary.

        Args:
            model_output (dict): Model output used by this operation.

        Returns:
            dict: Named result fields and metadata produced by this operation.
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
        """Compute the mode value for each field in a single file.

        Args:
            file_data (list): File data used by this operation.

        Returns:
            dict: Named result fields and metadata produced by this operation.
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
        """Entry point for null-value cleaning across all models and files.

        Iterates over all stored models and delegates cleaning of each
        file to `_clean_model_files`.

        Missing values are replaced using per-file mode statistics.

        Returns:
            None: This operation returns no value.
        """
        for model, files in self.models.items():
            self.__clean_model_files(model, files)

    def __clean_model_files(self, model: str, files: list) -> None:
        """Cleans all files for a given model.

        For each file:
        - Computes mode statistics based on file content
        - Applies cleaning logic to each item using `_clean_file`

        Args:
            model (str): Model used by this operation.
            files (list): Files used by this operation.

        Returns:
            None: This operation returns no value.
        """
        for file_index, file_data in enumerate(files):
            mode_dict = self.__compute_mode_for_file(file_data)
            self.__clean_file(model, file_index, file_data, mode_dict)

    def __clean_file(self, model: str, file_idx: int,
                    file_data: list, mode_dict: dict) -> None:
        """Cleans a single file dataset in-place.

        Iterates through all items in the file and replaces
        missing values using computed mode statistics.

        Args:
            model (str): Model used by this operation.
            file_idx (int): File idx used by this operation.
            file_data (list): File data used by this operation.
            mode_dict (dict): Mode dict used by this operation.

        Returns:
            None: This operation returns no value.
        """
        for current_index, item in enumerate(file_data):
            cleaned = self.__clean_item(item, mode_dict)

            if cleaned is not None:
                self.models[model][file_idx][current_index] = cleaned

    def __clean_item(self, item: Any, mode_dict: dict) -> Any:
        """Cleans a single data item.

        Rules:
        - If item is None → replace with mode dictionary copy
        - If item is a dict containing "answer" → clean only answer field
        - Otherwise → fill missing values directly in item dict

        Args:
            item (Any): Item used by this operation.
            mode_dict (dict): Mode dict used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
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
        """Cleans the "answer" sub-dictionary in-place.

        Replaces None values inside the answer dict with
        corresponding mode values.

        Args:
            answer (dict): Answer used by this operation.
            mode_dict (dict): Mode dict used by this operation.

        Returns:
            None: This operation returns no value.
        """
        if answer is None:
            return mode_dict.copy()

        for current_key, current_value in answer.items():
            if current_value is None:
                answer[current_key] = mode_dict[current_key]

    def __fill_missing_values(self, item: dict, mode_dict: dict) -> dict:
        """Fills missing (None) values in a dictionary in-place.

        Iterates over all keys and replaces None values using
        corresponding entries from mode_dict.

        Args:
            item (dict): Item used by this operation.
            mode_dict (dict): Mode dict used by this operation.

        Returns:
            dict: Named result fields and metadata produced by this operation.
        """
        for current_key, current_value in item.items():
            if current_value is None:
                item[current_key] = mode_dict[current_key]
        return item

    # ---------- STATS ----------
    def __compute_stats(self) -> pd.DataFrame:
        """Compute mean and standard deviation across files for each model.

        Returns:
            pd.DataFrame: The computed result for the supplied inputs.
        """
        all_results = []

        for model, files in self.models.items():

            lengths = [len(filename) for filename in files]
            if len(set(lengths)) != 1:
                raise ValueError(f"Inconsistent lengths in model {model}")

            num_points = lengths[0]

            first_item = self.__extract_scores(files[0][0])
            if first_item is None:
                raise ValueError("First item contains no valid scores")
            keys = list(first_item.keys())

            data = np.array([
                [
                    [self.__extract_scores(item)[current_key] for current_key in keys]
                    for item in file_data
                ]
                for file_data in files
            ])

            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            for current_index in range(num_points):
                row = {
                    "model": model,
                    "index": current_index
                }

                for column_index, current_key in enumerate(keys):
                    row[f"{current_key}_mean"] = mean[current_index, column_index]
                    row[f"{current_key}_std"] = std[current_index, column_index]

                all_results.append(row)

        return pd.DataFrame(all_results)

    # ---------- SAVE ----------
    def __save_to_csv(self, output_path: str) -> None:
        """Save computed statistics to a CSV file.

        Args:
            output_path (str): Output path used by this operation.

        Returns:
            None: This operation returns no value.
        """
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, PATHS["files"]["llm_aggregate"])

        data_frame = self.__compute_stats()
        data_frame.to_csv(output_path, index=False)

    def __save_all_to_csv(self, output_path: str) -> None:
        """Save all to csv.

        Args:
            output_path (str): Output path used by this operation.

        Returns:
            None: This operation returns no value.
        """
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, PATHS["files"]["llm_aggregate"])

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

                    for current_key, current_value in scores.items():
                        row[current_key] = current_value

                    rows.append(row)

        data_frame = pd.DataFrame(rows)
        data_frame.to_csv(output_path, index=False)

    # ---------- RUN ----------
    def run(self, output_path: str = PATHS["results"]["llm_aggregated"]) -> None:
        """Execute one named stage under the original request deadline and record its duration.

        Args:
            output_path (str): Output path used by this operation.

        Returns:
            None: This operation returns no value.
        """
        self.models.clear()
        self.__load_files()
        items = [item for files in self.models.values() for run in files for item in run]
        if any(isinstance(item, dict) and 'essay_id' in item for item in items):
            rows = []
            reference_ids, reference_manifest = None, None
            for model, files in self.models.items():
                for run_id, run in enumerate(files):
                    if not all(isinstance(item, dict) and item.get('essay_id') and item.get('dataset_manifest') for item in run):
                        raise ValueError('Cannot mix identified and positional predictions')
                    ids = [item['essay_id'] for item in run]
                    manifests = {item['dataset_manifest'] for item in run}
                    if len(ids) != len(set(ids)) or len(manifests) != 1:
                        raise ValueError('Duplicate essay IDs or mixed dataset releases')
                    if reference_ids is None:
                        reference_ids, reference_manifest = set(ids), manifests
                    if set(ids) != reference_ids or manifests != reference_manifest:
                        raise ValueError('Prediction runs belong to different samples or releases')
                    rows.extend({'model': model, 'file_id': run_id, 'sample_id': item['essay_id'],
                                 'dataset_manifest': item['dataset_manifest'],
                                 **legacy_prediction(item.get('answer'))} for item in run)
            destination = os.path.join(output_path, PATHS['files']['llm_aggregate']) if os.path.isdir(output_path) else output_path
            pd.DataFrame(rows).to_csv(destination, index=False)
            return
        self.__clean_nulls()
        self.__save_all_to_csv(output_path)
