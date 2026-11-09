# Tests

## Run

With the test dependencies installed, run the portable suite from the repository root.

```sh
python Manage.py test --suite portable
```

It uses local fixtures to check scoring, model output, retrieval errors and evaluation. See [Verification](../docs/guides/07Verification.md) for the full test suite and database checks.

## Find a test

- [`TestScoringService.py`](TestScoringService.py) covers an essay submission from input to scores.
- [`TestJudge.py`](TestJudge.py) and [`TestUILocalization.py`](TestUILocalization.py) cover judge replies and UI language switching.
- [`TestDatasetIntegration.py`](TestDatasetIntegration.py), [`TestPandaCorpus.py`](TestPandaCorpus.py) and [`TestDatabase.mjs`](TestDatabase.mjs) cover data preparation, the reference set and search.
- [`TestDistribution.py`](TestDistribution.py) and the [Kaggle tests](kaggle/README.md) check the source package and notebook setup.
