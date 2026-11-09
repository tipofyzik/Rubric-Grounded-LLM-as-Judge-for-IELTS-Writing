# Data

The project combines scored essays from Chillies and Writing9 into one dataset. This part cleans the source data, removes duplicates and keeps the essays for training and testing separate.

From the repository root, install the [research environment](../README.md). Choose the starting point that matches your files.

- **Original source files.** Place the Chillies train/test CSVs and Writing9 train/test JSONL files in the directories listed in [`paths.yaml`](../../shared/paths.yaml). Set `tasks.transform_datasets: 1` in [`config.yaml`](../config.yaml) to build the prepared release. This writes to `datasets.prepared`, so use a fresh destination when keeping an existing release.
- **Prepared release.** Place the verified release at `datasets.prepared` in [`paths.yaml`](../../shared/paths.yaml). Keep `tasks.transform_datasets: 0`, the default. The loader checks the manifest and file hashes before opening the splits.

```sh
python Manage.py research data --prepare-only
```

`--prepare-only` skips model runs. The default release directory is `data/datasets/processed/ielts-v3/`. It contains the train and test Parquet files, the manifest and audit reports. The command prints the verified split sizes and the path to the run directory containing `experiment_manifest.json`. Set `transform_datasets` to `0` before training AES.

[`PrepareData.py`](../PrepareData.py) starts the process. [`DatasetTransformer.py`](DatasetTransformer.py) reads and cleans the sources, [`Selection.py`](Selection.py) picks essays, and [`Writer.py`](Writer.py) saves the result. [`Analysis.py`](Analysis.py) summarizes the prepared data.
