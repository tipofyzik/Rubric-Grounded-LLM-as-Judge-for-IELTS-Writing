# Language model experiments

The [legacy Kaggle notebook](../notebooks/KaggleNotebook.ipynb) uses these modules to fine-tune models and collect their replies.

- [`LLMFineTuner.py`](LLMFineTuner.py) trains a language model on scored essays.
- [`LLMResponseObtainer.py`](LLMResponseObtainer.py) sends prompts to a local or hosted model and saves its replies.
