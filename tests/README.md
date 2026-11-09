# Tests

The portable suite checks scoring, judge output, retrieval recovery, evaluation and saved LLM replies with small local fixtures and models. It runs on CPU without credentials, source datasets or pretrained weights.

With Python 3.11, run these commands from the repository root. The research profile installs the CPU dependencies in the project’s `.venv`. CI uses the same test command with its [pinned dependency list](../.github/workflows/demo-checks.yml).

```sh
python Manage.py install research
python Manage.py test --suite portable
```

A successful run exits with code 0 and reports zero failures and errors. It writes a summary to `.cache/checks/results.json` and a log to `.cache/checks/checks.log`.

| Suite | Coverage |
| --- | --- |
| `portable` | The five modules listed in [`RunPortable.py`](RunPortable.py). |
| `main` | Python regression tests, including UI localization, dataset preparation, the PANDA corpus and source packaging. Install the local profile for Gradio as well. |
| `legacy` | [Local tests for the historical Kaggle notebook](kaggle/README.md). The runner uses samples from the configured source files. |
| `deployment` | Local source-build and runtime checks for the hosted demo. Install the public operator profile for these checks. |
| `all` | Runs `main`, `legacy` and `deployment` in sequence. Use the configured fixtures and all required dependency profiles. |

## SQL checks

With Node.js and the project Python environment active, generate a local fixture and run the PGlite harness separately.

```sh
npm install --prefix .cache/sql --cache .cache/npm --no-audit --no-fund @electric-sql/pglite@0.5.8 @electric-sql/pglite-pgvector@0.0.9
python -B tests/TestPandaCorpus.py --sql-fixture .cache/sql-fixture
node --preserve-symlinks --preserve-symlinks-main tests/TestDatabase.mjs .cache/sql-fixture
```

The harness applies the migrations to its local database and prints the check results. [Verification and release](../docs/guides/Verify.md)
