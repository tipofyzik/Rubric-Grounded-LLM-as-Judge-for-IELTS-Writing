# Prepare data

## Download

Install the [research environment](../../code/offline_research/README.md) and activate the project `.venv` for the `hf` command. Run from the repository root. The revisions below match the four source files used for the published benchmark.

```sh
hf download chillies/IELTS-writing-task-2-evaluation train.csv test.csv --repo-type dataset --revision ce4ab2f4e652ce08582f5488b23e1ebffa222e26 --local-dir "data/datasets/chillies - IELTS-writing-task-2-evaluation"
hf download ndtran0101/writing9-ielts-essays data/train.jsonl data/test.jsonl --repo-type dataset --revision 07984374ff6dc6738b126dc2b861f5b741ef461a --local-dir "data/datasets/ndtran0101 - writing9-ielts-essays"
```

In the Chillies directory, rename `train.csv` to `hugging_face_train.csv` and `test.csv` to `hugging_face_test.csv`. [Input paths](../../code/shared/paths.yaml)

## Build

Set `tasks.transform_datasets: 1` in [`config.yaml`](../../code/offline_research/config.yaml). The baseline uses `dataset.source: combined`, seed 42 and per-group quotas of 500 for train and 100 for test. To keep an existing release, set `datasets.prepared` to a new output directory in [`paths.yaml`](../../code/shared/paths.yaml).

```sh
python Manage.py research data --prepare-only
```

The command writes train and test Parquet files, the manifest and audit reports to `data/datasets/processed/ielts-v3/` by default. The baseline contains 8,236 train and 1,635 test essays. The command also saves the experiment selection. Set `transform_datasets` back to `0` to load the prepared release on subsequent runs.

The manifest records source paths and dependency versions, so a rebuild can produce a different SHA-256 with the same essays and labels. The published benchmark uses the original compatible dataset, AES and PANDA releases. [Benchmark provenance](../model_evaluation_protocol.md#accepted-baseline-and-provenance)
