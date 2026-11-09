# Tests

These checks cover what the app returns, how it handles errors and whether the research tools can use the same data. The portable suite uses small local examples and models.

With the test dependencies installed, run this from the repo root.

```sh
python Manage.py test --suite portable
```

- [`TestScoringService.py`](TestScoringService.py) checks the path from an essay to its scores.
- [`TestUILocalization.py`](TestUILocalization.py) checks what happens when users switch languages.
- [`TestDatasetIntegration.py`](TestDatasetIntegration.py) checks data preparation. [`TestPandaCorpus.py`](TestPandaCorpus.py) checks the collection of scored examples.
- [`TestDistribution.py`](TestDistribution.py) checks the source package. [Kaggle tests](kaggle/README.md) cover the notebook setup.

[Full test suite](../docs/guides/07Verification.md)
