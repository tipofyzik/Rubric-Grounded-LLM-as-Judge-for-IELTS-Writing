# Kaggle tests

These checks run code from the historical [`KaggleNotebook.ipynb`](../../code/offline_research/notebooks/KaggleNotebook.ipynb) locally with small example files. They check that the notebook can open its inputs and save the expected results.

Install the research profile and place the original source files at the [configured dataset paths](../../code/shared/paths.yaml). Run `python Manage.py test --suite legacy` from the repository root. The runner builds a small fixture and writes `Results.json` and `Checks.log` under `code/offline_research/notebooks/artifacts/tests/`.

[`TestKaggleNotebook.py`](TestKaggleNotebook.py) covers notebook setup and output files. [`TestLLMResponseObtainer.py`](TestLLMResponseObtainer.py) covers model requests and saved replies.

[Kaggle notebooks](../../code/offline_research/notebooks/README.md)
