# Data

Prepare `ielts-v3` from Chillies and Writing9. The pipeline filters invalid scores, removes duplicate essays and preserves the train and test splits.

To build from source files, follow [Prepare data](../../../docs/guides/PrepareData.md). To use an existing prepared release, place it under `data/datasets/processed/ielts-v3/` and run this from the repository root with the default configuration.

```sh
python Manage.py research data --prepare-only
```

The command prints the split sizes and saves the experiment selection.

[`PrepareData.py`](../PrepareData.py) · [Dataset preparation](DatasetTransformer.py) · [Selection](Selection.py)
