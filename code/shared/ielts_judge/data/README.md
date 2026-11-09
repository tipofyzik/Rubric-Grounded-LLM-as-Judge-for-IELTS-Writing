# Data

Training and evaluation use these helpers to read the same dataset and keep track of which essays they use.

The prepared input contains `train.parquet`, `test.parquet` and a manifest with hashes for the release files. The loader verifies the files and checks that essay IDs and normalized essay hashes are disjoint across the splits. The experiment manifest records which test IDs were selected and ties them to the prepared release.

[`Preparation.py`](Preparation.py) reads the source and prepared data. [`Experiment.py`](Experiment.py) selects the essays used in an experiment and records which ones were selected.

[Data preparation](../../../offline_research/data/README.md)
