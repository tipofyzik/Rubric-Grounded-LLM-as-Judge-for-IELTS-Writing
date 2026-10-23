"""One offline entry point for Chillies extraction and Writing9 preparation."""

import re
import pandas as pd
from ..criteria import CRITERION_IDS, IELTS_CRITERIA, LEGACY_TO_PUBLIC


class DatasetTransformer:
    def __init__(self, datasets_paths, selection=None, *, save_intermediate_datasets=False):
        self.datasets_paths = datasets_paths
        self.selection = selection
        self.save_intermediate_datasets = save_intermediate_datasets

    def prepare(self, *, limit_per_split=None):
        """Extract both local sources and resolve their duplicates together."""
        from .preparation import PreparedDataset, extract_sources, resolve_duplicates, source_paths
        paths = source_paths(self.datasets_paths["source_root"])
        for files in paths.values():
            for path in files.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
        extracted, sources = extract_sources(paths, self.extract_score, limit_per_split)
        return PreparedDataset(resolve_duplicates(extracted), sources, extracted)


    def save_processed(self, prepared, output):
        from .preparation import save_prepared, source_paths
        return save_prepared(prepared, output, source_paths(self.datasets_paths["source_root"]),
                             selection=self.selection, save_intermediate_datasets=self.save_intermediate_datasets)

    def prepare_from_extracted(self, release, *, allow_partial=False):
        """Reapply duplicate/cleanup rules to verified extracted rows, without regex parsing."""
        from pathlib import Path
        from .preparation import PreparedDataset, file_hash, resolve_duplicates
        _, _, manifest = self.load_processed(release, allow_partial=allow_partial)
        release = Path(release)
        extracted = {source: pd.read_parquet(release / 'extracted' / f'{source}.parquet')
                     for source in ('chillies', 'writing9')}
        return PreparedDataset(resolve_duplicates(extracted), manifest['sources'], extracted,
                               parent_manifest=file_hash(release / 'manifest.json'))


    def prepare_and_save(self, output, *, limit_per_split=None):
        from .preparation import source_paths, validate_output
        validate_output(output, source_paths(self.datasets_paths["source_root"]))
        return self.save_processed(self.prepare(limit_per_split=limit_per_split), output)


    @staticmethod
    def load_processed(output, source="combined", *, allow_partial=False):
        from .preparation import load_prepared
        return load_prepared(output, source, allow_partial=allow_partial)


    def extract_score(self,text: str, criterion: str, next_criterion: str | None = None) -> float|None:
        """
        Extracts the score for a given criterion from the evaluation text.

        Args:
            text (str): The evaluation text containing the scores.
            criterion (str): The criterion for which to extract the score.
            next_criterion (str | None): The next criterion in the evaluation text, used to define the block of text to search within. If None, the search will go until the end of the text.
        Returns:

            float | None: The extracted score for the given criterion, or None if no score is
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


    def load_hugging_face_datasets(self):
        """Read original CSVs with source split/row provenance; never read a cache."""
        from .preparation import source_paths
        files = source_paths(self.datasets_paths['source_root'])['chillies']
        frames = []
        for split, path in files.items():
            frame = pd.read_csv(path)
            frame['source_split'] = split
            frame['source_line'] = range(1, len(frame) + 1)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def transform_hugging_face_dataset(self):
        """Re-extract old evaluations without rewriting source CSVs or requiring Writing9."""
        from .preparation import extract_sources, source_paths
        paths = source_paths(self.datasets_paths['source_root'])
        extracted, _ = extract_sources({'chillies': paths['chillies']}, self.extract_score)
        return extracted['chillies']

    def load_transformed_hugging_face_dataset(self):
        """Compatibility view of accepted Chillies Parquet rows, with legacy names."""
        train, test, _ = self.load_processed(self.datasets_paths['prepared'], 'chillies')
        mapping = {v:k for k,v in LEGACY_TO_PUBLIC.items()}
        mapping.update(task_prompt='prompt', overall_label='band')
        return pd.concat([train, test], ignore_index=True).rename(columns=mapping)

    def fill_missing_scores(self, dataset):
        """Missing labels remain unknown; overall is never an imputation source."""
        return dataset
