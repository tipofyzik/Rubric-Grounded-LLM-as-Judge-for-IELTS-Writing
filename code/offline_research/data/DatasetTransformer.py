"""One offline entry point for Chillies extraction and Writing9 preparation."""
from typing import Any

import re
import pandas as pd
from ielts_judge.Criteria import CRITERION_IDS, IELTS_CRITERIA, LEGACY_TO_PUBLIC


class DatasetTransformer:
    """Manage dataset transformer through the operations defined by this component."""
    def __init__(self, datasets_paths: Any, selection: Any=None, *, save_intermediate_datasets: bool=False) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            datasets_paths (Any): Datasets paths used by this operation.
            selection (Any): Selection used by this operation.
            save_intermediate_datasets (bool): Save intermediate datasets used by this operation.

        Returns:
            None: This operation returns no value.
        """
        self.datasets_paths = datasets_paths
        self.selection = selection
        self.save_intermediate_datasets = save_intermediate_datasets

    def prepare(self, *, limit_per_split: Any=None) -> Any:
        """Assemble the current operation's validated inputs and metadata.

        Args:
            limit_per_split (Any): Limit per split used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from ielts_judge.data.Preparation import PreparedDataset, extract_sources, resolve_duplicates, source_paths
        paths = source_paths(self.datasets_paths["source_root"])
        for files in paths.values():
            for path in files.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
        extracted, sources = extract_sources(paths, self.extract_score, limit_per_split)
        return PreparedDataset(resolve_duplicates(extracted), sources, extracted)


    def save_processed(self, prepared: Any, output: Any) -> Any:
        """Save processed.

        Args:
            prepared (Any): Prepared used by this operation.
            output (Any): Output used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from ielts_judge.data.Preparation import source_paths
        from offline_research.data.Writer import save_prepared
        return save_prepared(prepared, output, source_paths(self.datasets_paths["source_root"]),
                             selection=self.selection, save_intermediate_datasets=self.save_intermediate_datasets)

    def prepare_from_extracted(self, release: Any, *, allow_partial: bool=False) -> Any:
        """Reapply duplicate/cleanup rules to verified extracted rows, without regex parsing.

        Args:
            release (Any): Release used by this operation.
            allow_partial (bool): Allow partial used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from pathlib import Path
        from ielts_judge.data.Preparation import PreparedDataset, file_hash, resolve_duplicates
        _, _, manifest = self.load_processed(release, allow_partial=allow_partial)
        release = Path(release)
        extracted = {source: pd.read_parquet(release / 'extracted' / f'{source}.parquet')
                     for source in ('chillies', 'writing9')}
        return PreparedDataset(resolve_duplicates(extracted), manifest['sources'], extracted,
                               parent_manifest=file_hash(release / 'manifest.json'))


    def prepare_and_save(self, output: Any, *, limit_per_split: Any=None) -> Any:
        """Prepare and save.

        Args:
            output (Any): Output used by this operation.
            limit_per_split (Any): Limit per split used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from ielts_judge.data.Preparation import source_paths
        from offline_research.data.Writer import validate_output
        validate_output(output, source_paths(self.datasets_paths["source_root"]))
        return self.save_processed(self.prepare(limit_per_split=limit_per_split), output)


    @staticmethod
    def load_processed(output: Any, source: str="combined", *, allow_partial: bool=False) -> Any:
        """Load processed.

        Args:
            output (Any): Output used by this operation.
            source (str): Source used by this operation.
            allow_partial (bool): Allow partial used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from ielts_judge.data.Preparation import load_prepared
        return load_prepared(output, source, allow_partial=allow_partial)


    def extract_score(self,text: str, criterion: str, next_criterion: str | None = None) -> float|None:
        """Extracts the score for a given criterion from the evaluation text.

        Args:
            text (str): Text used by this operation.
            criterion (str): Criterion used by this operation.
            next_criterion (str | None): Next criterion used by this operation.

        Returns:
            float | None: The computed result for the supplied inputs.
        """
        # Define the regex pattern to extract the block of text for the given criterion
        if next_criterion:
            pattern_block = rf"({re.escape(criterion)}.*?)(?={re.escape(next_criterion)})"
        else:
            pattern_block = rf"({re.escape(criterion)}.*)"

        match_block = re.search(pattern_block, text, flags=re.DOTALL)
        if not match_block:
            pattern_block = rf"({re.escape(criterion)}.*)"
            match_block = re.search(pattern_block, text, flags=re.DOTALL)
            if not match_block:
                return None
        block_text = match_block.group(1)

        # First, try to find a line that contains the criterion name
        # Do not add re.IGNORECASE flag to the regex pattern, because it will cause problems
        # in cases when criterion additionally mentioned in the evaluation itself.
        match_criterion_line = re.search(rf"^.*{re.escape(criterion)}.*$", block_text, flags=re.MULTILINE)
        if match_criterion_line:
            # In case if the overall band goes first but not the last, we start
            # from the criterion line and search tll the end of the block, not the next criterion line
            # It fixes the problem of not getting the score for the last criterion when the overall band goes first.
            line = match_criterion_line.group(0)
            match_number = re.search(r"\b(\d+(?:\.\d+)?)\b", line)
            if match_number:
                return float(match_number.group(1))

        # If no line contains the criterion name, try to find lines that contain the word "score"
        # Same warning about re.IGNORECASE flag as above!
        score_lines = re.findall(r"^.*Score.*$", block_text, flags=re.MULTILINE)
        if score_lines:
            for line in score_lines:
                match_number = re.search(r"(\d+(?:\.\d+)?)", line)
                if match_number:
                    return float(match_number.group(1))

        return None


    def load_hugging_face_datasets(self) -> Any:
        """Read original CSVs with source split/row provenance; never read a cache.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from ielts_judge.data.Preparation import source_paths
        files = source_paths(self.datasets_paths['source_root'])['chillies']
        frames = []
        for split, path in files.items():
            frame = pd.read_csv(path)
            frame['source_split'] = split
            frame['source_line'] = range(1, len(frame) + 1)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def transform_hugging_face_dataset(self) -> Any:
        """Re-extract old evaluations without rewriting source CSVs or requiring Writing9.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from ielts_judge.data.Preparation import extract_sources, source_paths
        paths = source_paths(self.datasets_paths['source_root'])
        extracted, _ = extract_sources({'chillies': paths['chillies']}, self.extract_score)
        return extracted['chillies']

    def load_transformed_hugging_face_dataset(self) -> Any:
        """Compatibility view of accepted Chillies Parquet rows, with legacy names.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        train, test, _ = self.load_processed(self.datasets_paths['prepared'], 'chillies')
        mapping = {current_value:current_key for current_key,current_value in LEGACY_TO_PUBLIC.items()}
        mapping.update(task_prompt='prompt', overall_label='band')
        return pd.concat([train, test], ignore_index=True).rename(columns=mapping)

    def fill_missing_scores(self, dataset: Any) -> Any:
        """Missing labels remain unknown; overall is never an imputation source.

        Args:
            dataset (Any): Dataset used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        return dataset
