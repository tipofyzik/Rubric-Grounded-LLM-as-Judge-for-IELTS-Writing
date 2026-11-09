# Train the essay scoring model

This part trains the model that makes the app’s first score estimate. It learns from scored essays and is also tested on its own in the model comparison.

Start at [`TrainAes.py`](../TrainAes.py) to train the model or use a saved one. [`Training.py`](Training.py) handles training and saves the model. [`RegressorEvaluator.py`](RegressorEvaluator.py) compares its predictions with the dataset scores.

[Kaggle training](../notebooks/README.md)
