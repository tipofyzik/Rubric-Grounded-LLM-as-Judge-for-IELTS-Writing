# First score estimate

This model reads the essay and estimates the four scores. The judge uses that estimate as one input. In the research, the same model is used on its own for comparison.

[`Inference.py`](Inference.py) loads the trained model and scores essays. [`OrdinalRegression.py`](OrdinalRegression.py) defines the model.

[Training](../../../offline_research/aes/README.md)
