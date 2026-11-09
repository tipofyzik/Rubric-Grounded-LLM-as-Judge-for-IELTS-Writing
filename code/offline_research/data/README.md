# Prepare the dataset

[`PrepareData.py`](../PrepareData.py) is the entry point. It prepares a common dataset from Chillies and Writing9 for training and evaluation.

- [`DatasetTransformer.py`](DatasetTransformer.py) reads and cleans the source essays.
- [`Profiles.py`](Profiles.py) defines score groups. [`Selection.py`](Selection.py) chooses essays using those groups.
- [`Writer.py`](Writer.py) saves the train and test splits. [`Runtime.py`](Runtime.py) turns those rows into model inputs.
- [`Analysis.py`](Analysis.py) and [`RawDataAnalyzer.py`](RawDataAnalyzer.py) summarize the prepared and source data.
