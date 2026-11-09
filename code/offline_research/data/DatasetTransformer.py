"""One offline entry point for Chillies extraction and Writing9 preparation."""
from typing import Any

import re
import pandas as pd
from ielts_judge.Criteria import CRITERION_IDS, IELTS_CRITERIA, LEGACY_TO_PUBLIC


class DatasetTransformer:
    """Prepare source datasets and read or write verified split releases."""
    def __init__(self, datasets_paths: Any, selection: Any=None, *, save_intermediate_datasets: bool=False) -> None:
        """Configure dataset input paths and release selection.

        Args:
            datasets_paths (Any): Dataset path mapping with source_root and prepared locations.
            selection (Any): Optional policy for filtering score profiles and selecting rows.
            save_intermediate_datasets (bool): Include extracted and excluded rows in new releases.
        """
        self.datasets_paths = datasets_paths
        self.selection = selection
        self.save_intermediate_datasets = save_intermediate_datasets

    def prepare(self, *, limit_per_split: Any=None) -> Any:
        """Extract the source rows and resolve duplicates in memory.

        Args:
            limit_per_split (Any): Optional row cap per raw file for a partial test release.

        Returns:
            Any: PreparedDataset with cleaned rows, source metadata and extracted rows.

        Raises:
            FileNotFoundError: A required source file is missing.
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
        """Write or rebuild a prepared release with the configured selection policy.

        Args:
            prepared (Any): PreparedDataset produced by prepare or prepare_from_extracted.
            output (Any): Release directory. A rebuild replaces the published split files.

        Returns:
            Any: Path to the release directory.
        """
        from ielts_judge.data.Preparation import source_paths
        from offline_research.data.Writer import save_prepared
        return save_prepared(prepared, output, source_paths(self.datasets_paths["source_root"]),
                             selection=self.selection, save_intermediate_datasets=self.save_intermediate_datasets)

    def prepare_from_extracted(self, release: Any, *, allow_partial: bool=False) -> Any:
        """Reapply cleanup to the saved extracted rows of a verified release.

        Args:
            release (Any): Release directory containing both extracted source Parquet files.
            allow_partial (bool): Accept a release that declares partial source input.

        Returns:
            Any: PreparedDataset with cleaned rows and the parent manifest identity.
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
        """Extract and clean source rows, then write a prepared dataset release.

        Args:
            output (Any): Release directory outside the raw source inputs.
            limit_per_split (Any): Optional row cap per raw file for a partial test release.

        Returns:
            Any: Path to the written release.
        """
        from ielts_judge.data.Preparation import source_paths
        from offline_research.data.Writer import validate_output
        validate_output(output, source_paths(self.datasets_paths["source_root"]))
        return self.save_processed(self.prepare(limit_per_split=limit_per_split), output)


    @staticmethod
    def load_processed(output: Any, source: str="combined", *, allow_partial: bool=False) -> Any:
        """Read a prepared release through the shared dataset validator.

        Args:
            output (Any): Existing release directory containing the manifest and splits.
            source (str): Keep both sources (combined), or filter to chillies or writing9.
            allow_partial (bool): Allow a release that declares partial source input.

        Returns:
            Any: (train DataFrame, test DataFrame, manifest dict).
        """
        from ielts_judge.data.Preparation import load_prepared
        return load_prepared(output, source, allow_partial=allow_partial)


    def extract_score(self,text: str, criterion: str, next_criterion: str | None = None) -> float|None:
        """Read one criterion score from a Chillies evaluation response.

        Args:
            text (str): Original evaluation response.
            criterion (str): Case-sensitive criterion heading.
            next_criterion (str | None): Next heading, used to bound the search when present.

        Returns:
            float | None: First parsed score from the criterion line or a Score line,
                or None if no score is found.
        """
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

        # Match the source heading's case so mentions in the prose do not win.
        match_criterion_line = re.search(rf"^.*{re.escape(criterion)}.*$", block_text, flags=re.MULTILINE)
        if match_criterion_line:
            line = match_criterion_line.group(0)
            match_number = re.search(r"\b(\d+(?:\.\d+)?)\b", line)
            if match_number:
                return float(match_number.group(1))

        # Some responses put the score on a separate, capitalized Score line.
        score_lines = re.findall(r"^.*Score.*$", block_text, flags=re.MULTILINE)
        if score_lines:
            for line in score_lines:
                match_number = re.search(r"(\d+(?:\.\d+)?)", line)
                if match_number:
                    return float(match_number.group(1))

        return None


    def load_hugging_face_datasets(self) -> Any:
        """Read local Chillies source CSVs and attach row provenance.

        Returns:
            Any: DataFrame with original CSV columns plus source_split and source_line.
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
        """Extract labels and text from the local Chillies sources.

        Returns:
            Any: DataFrame of extracted Chillies rows before cross-source deduplication.
        """
        from ielts_judge.data.Preparation import extract_sources, source_paths
        paths = source_paths(self.datasets_paths['source_root'])
        extracted, _ = extract_sources({'chillies': paths['chillies']}, self.extract_score)
        return extracted['chillies']

    def load_transformed_hugging_face_dataset(self) -> Any:
        """Read both prepared Chillies splits through the legacy column view.

        Returns:
            Any: Combined DataFrame with legacy criterion, prompt and band column names.
        """
        train, test, _ = self.load_processed(self.datasets_paths['prepared'], 'chillies')
        mapping = {current_value:current_key for current_key,current_value in LEGACY_TO_PUBLIC.items()}
        mapping.update(task_prompt='prompt', overall_label='band')
        return pd.concat([train, test], ignore_index=True).rename(columns=mapping)

    def fill_missing_scores(self, dataset: Any) -> Any:
        """Keep missing scores unchanged for legacy callers.

        Args:
            dataset (Any): Dataset whose original labels are kept.

        Returns:
            Any: The input object unchanged.
        """
        return dataset
