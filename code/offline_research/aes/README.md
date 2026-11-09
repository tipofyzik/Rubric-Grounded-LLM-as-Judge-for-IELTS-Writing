# AES

Train the essay-only model used for preliminary scores and as a benchmark baseline.

Use a [prepared dataset](../data/README.md), the default configuration and a new output directory. Run from the repository root.

```sh
python Manage.py research aes train --output results/aes/runs/my-run
python Manage.py research aes predict --bundle results/aes/runs/my-run/aes_bundle --essay-file essay.txt
```

Training saves the model under `aes_bundle/` and writes test predictions and metrics. Prediction takes a UTF-8 essay and prints JSON.

[`TrainAes.py`](../TrainAes.py) · [Kaggle](../notebooks/README.md) · [Release layout](../../../docs/guides/Install.md#release-layout)
