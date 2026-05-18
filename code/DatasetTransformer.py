import pandas as pd
import re



class DatasetTransformer:
    """
    This class is responsible for transforming the dataset, including feature engineering and selection.
    """
    def __init__(self, datasets_paths: dict[str, str]):
        """
        Initializes the DatasetTransformer instance.

        Arguments:
            hugging_face_dataset_path (str): The file path to the Hugging Face dataset.
            kaggle_dataset_path (str): The file path to the Kaggle dataset.
            output_path (str): The file path where the output dataset will be saved.
        """
        self.datasets_paths = datasets_paths
    

    # Loading datasets
    def load_hugging_face_datasets(self) -> pd.DataFrame:
        """
        Loads the Hugging Face datasets from the specified file paths.
        Returns:
            pandas.DataFrame: A concatenated DataFrame containing both the Hugging Face train and test datasets.
        """
        hugging_face_train_dataset = pd.read_csv(self.datasets_paths["hugging_face_train_dataset_path"])
        hugging_face_test_dataset = pd.read_csv(self.datasets_paths["hugging_face_test_dataset_path"])
        self.hugging_face_dataset = pd.concat([hugging_face_train_dataset, hugging_face_test_dataset], ignore_index=True)
        return self.hugging_face_dataset

    def load_kaggle_dataset(self) -> pd.DataFrame:
        """
        Loads the Kaggle dataset from the specified file path.
        Returns:
            pandas.DataFrame: A DataFrame containing the Kaggle dataset.
        """
        self.kaggle_dataset = pd.read_csv(self.datasets_paths["kaggle_dataset_path"])
        return self.kaggle_dataset

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
    
    def fill_missing_scores(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing scores in the dataset using the mean score for each criterion.

        Args:
            dataset (pd.DataFrame): The dataset with missing scores.
        """
        null_rows = dataset[dataset.isnull().any(axis=1)]
        if null_rows.empty:
            print("The Hugging Face dataset does not have any null values.")
            return dataset
        
        for idx, row in null_rows.iterrows():
            raw_value = str(dataset.at[idx, "band"])
            match = re.search(r"\d+(?:\.\d+)?", raw_value)
            band_value = float(match.group()) if match else None
            dataset.loc[idx,
                        ["task_achievement", 
                        "coherence_and_cohesion", 
                        "lexical_resource", 
                        "grammatical_range_and_accuracy"]
                        ] = band_value
        return dataset

    def transform_hugging_face_dataset(self) -> None:
        """
        Transforms the Hugging Face dataset by performing feature engineering and selection.
        Returns:
            pandas.DataFrame: A transformed DataFrame containing the Hugging Face dataset.
        """
        self.hugging_face_dataset = self.load_hugging_face_datasets()
        criteria = ["Task Achievement",
                    "Coherence and Cohesion",
                    "Lexical Resource",
                    "Grammatical Range and Accuracy",
                    "Overall Band"
                    ]
        for i, criterion in enumerate(criteria[:4]):
            # Define the next criterion for block delimitation, or None if this is the last criterion
            next_criterion = criteria[i + 1] if i + 1 < len(criteria) else None

            # Extract the score for the current criterion and create a new column in the dataset
            self.hugging_face_dataset[criterion.lower().replace(" ", "_")] = \
                self.hugging_face_dataset["evaluation"].apply(
                    lambda x, criterion=criterion, next_criterion=next_criterion: self.extract_score(x, criterion, next_criterion)
                )
        
        self.hugging_face_dataset = self.fill_missing_scores(self.hugging_face_dataset) 

        self.hugging_face_dataset.drop(index = 1579, inplace=True) # dropping the row with the evaluation that does not contain essay.
        # --- Saving final dataset ---
        self.hugging_face_dataset.to_csv(self.datasets_paths["transformed_hugging_face_dataset_path"], index=False)

    def load_transformed_hugging_face_dataset(self) -> pd.DataFrame:
        """
        Loads the transformed Hugging Face dataset from the specified file path.
        Returns:
            pandas.DataFrame: A DataFrame containing the transformed Hugging Face dataset.
        """
        self.transformed_hugging_face_dataset = pd.read_csv(self.datasets_paths["transformed_hugging_face_dataset_path"])
        return self.transformed_hugging_face_dataset