# AES

This part trains the model that makes the app’s first score estimate. It learns from scored essays and is also tested on its own in the model comparison.

Train from the repository root with a verified [prepared dataset](../data/README.md), `transform_datasets: 0` and a new output directory. Training initializes the encoder from the pinned model in [`config.yaml`](../config.yaml).

```sh
python Manage.py research aes train --output results/aes/runs/my-run
python Manage.py research aes predict --bundle results/aes/runs/my-run/aes_bundle --essay-file essay.txt
```

Training writes `aes_bundle/`, `aes_predictions.csv`, `aes_report.json` and `experiment_manifest.json`. Prediction takes one UTF-8 essay and prints its scores and token lengths as JSON. The app and benchmark use the bundle through a [release descriptor](../README.md#releases).

Start at [`TrainAes.py`](../TrainAes.py) to train the model or use a saved one. [`Training.py`](Training.py) handles training and saves the model. [`RegressorEvaluator.py`](RegressorEvaluator.py) compares its predictions with the dataset scores.

[Kaggle training](../notebooks/README.md)
