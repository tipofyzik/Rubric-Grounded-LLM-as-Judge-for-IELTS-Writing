# Prepare the dataset

The project combines scored essays from Chillies and Writing9 into one dataset. This part cleans the source data, removes duplicates and keeps the essays for training and testing separate.

[`PrepareData.py`](../PrepareData.py) starts the process. [`DatasetTransformer.py`](DatasetTransformer.py) reads and cleans the sources, [`Selection.py`](Selection.py) picks essays, and [`Writer.py`](Writer.py) saves the result. [`Analysis.py`](Analysis.py) summarizes the prepared data.
