# Research

The research code prepares the data used by the demo and compares ways to score an essay.

- [`data/`](data/README.md) cleans scored essays and builds the train and test splits.
- [`aes/`](aes/README.md) trains a model to estimate scores from the essay text.
- [`panda/`](panda/README.md) generates notes on scored training essays and builds the reference set.
- [`evaluation/`](evaluation/README.md) compares predictions with the scores in the test set.
- [`llm/`](llm/README.md) contains the fine-tuning and inference tools used in earlier experiments.

[Kaggle](notebooks/README.md) · [Results](../../docs/model_evaluation_report.md)
