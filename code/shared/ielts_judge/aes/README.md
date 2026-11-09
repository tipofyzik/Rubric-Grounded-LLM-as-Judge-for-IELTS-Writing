# Essay scoring

The automated essay scoring model (AES) gives the judge an initial estimate for each criterion. It uses the essay text alone.

- [`Inference.py`](Inference.py) loads the trained model and predicts scores.
- [`OrdinalRegression.py`](OrdinalRegression.py) defines the model.

[Training](../../../offline_research/aes/README.md)
