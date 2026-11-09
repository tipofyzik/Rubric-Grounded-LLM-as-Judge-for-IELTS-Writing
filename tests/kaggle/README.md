# Kaggle tests

Test the [legacy Kaggle notebook](../../code/offline_research/notebooks/KaggleNotebook.ipynb) locally.

Install the research profile and put the source files at the [configured dataset paths](../../code/shared/paths.yaml). From the repository root, run `python Manage.py test --suite legacy`.

[`TestKaggleNotebook.py`](TestKaggleNotebook.py) · [`TestLLMResponseObtainer.py`](TestLLMResponseObtainer.py)
