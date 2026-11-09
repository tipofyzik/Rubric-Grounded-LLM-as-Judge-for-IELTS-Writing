# Data

The project builds `ielts-v3` from Chillies and Writing9. It validates the scores, keeps the low-band Chillies examples, removes duplicate essays and applies score-profile quotas within the original train and test splits.

## Download

Download the four source files from the revisions below. Their contents match the source files used for the published benchmark. Install the [research environment](../README.md), activate the project `.venv` for the `hf` command and run from the repository root.

```sh
hf download chillies/IELTS-writing-task-2-evaluation train.csv test.csv --repo-type dataset --revision ce4ab2f4e652ce08582f5488b23e1ebffa222e26 --local-dir "data/datasets/chillies - IELTS-writing-task-2-evaluation"
hf download ndtran0101/writing9-ielts-essays data/train.jsonl data/test.jsonl --repo-type dataset --revision 07984374ff6dc6738b126dc2b861f5b741ef461a --local-dir "data/datasets/ndtran0101 - writing9-ielts-essays"
```

In the Chillies directory, rename `train.csv` to `hugging_face_train.csv` and `test.csv` to `hugging_face_test.csv`. The Writing9 files already match the expected layout. [Paths](../../shared/paths.yaml)

## Prepare

Set `tasks.transform_datasets: 1` in [`config.yaml`](../config.yaml) and run the command below. The baseline uses `dataset.source: combined`, seed 42 and per-group quotas of 500 for train and 100 for test. To preserve an existing release, choose a new `datasets.prepared` output directory in [`paths.yaml`](../../shared/paths.yaml).

```sh
python Manage.py research data --prepare-only
```

`--prepare-only` skips model runs. The command reads local files, writes the Parquet splits, manifest and audit reports to `data/datasets/processed/ielts-v3/` by default, and prints the split sizes and experiment directory.

Set `transform_datasets` back to `0` to use the prepared release. This is the default, and the loader verifies the manifest and file hashes. Rebuilding can change the manifest SHA-256 because the manifest records source paths and package versions. The published benchmark uses the exact accepted dataset, AES and PANDA releases. [Baseline provenance](../../../docs/model_evaluation_protocol.md#accepted-baseline-and-provenance)

[`PrepareData.py`](../PrepareData.py) starts the process. [`DatasetTransformer.py`](DatasetTransformer.py) reads and cleans the sources, [`Selection.py`](Selection.py) picks essays, and [`Writer.py`](Writer.py) saves the result. [`Analysis.py`](Analysis.py) summarizes the prepared data.
