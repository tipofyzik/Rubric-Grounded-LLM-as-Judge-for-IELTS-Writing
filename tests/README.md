# Tests

Run the CPU test suite from the repository root with Python 3.11. It uses small local fixtures.

```sh
python Manage.py install research
python Manage.py test --suite portable
```

Results and logs are written to `.cache/checks/`.

| Suite | Coverage |
| --- | --- |
| `portable` | Scoring, retrieval and evaluation checks. |
| `main` | Regression tests, including the UI. Install the local profile for Gradio. |
| `legacy` | [Local Kaggle notebook tests](kaggle/README.md). |
| `deployment` | Local source-build and runtime checks for the hosted demo. Install the public operator profile for these checks. |
| `all` | Runs `main`, `legacy` and `deployment` in sequence. Use the configured fixtures and all required dependency profiles. |

[SQL checks](../docs/guides/Verify.md#sql) · [CI](../.github/workflows/demo-checks.yml)
