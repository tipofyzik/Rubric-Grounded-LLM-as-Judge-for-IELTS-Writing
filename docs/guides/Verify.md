# Verify

[Python tests](../../tests/README.md)

## SQL

Run from the repository root with Node.js and the project Python environment active.

```sh
npm install --prefix .cache/sql --cache .cache/npm --no-audit --no-fund @electric-sql/pglite@0.5.8 @electric-sql/pglite-pgvector@0.0.9
python -B tests/TestPandaCorpus.py --sql-fixture .cache/sql-fixture
node --preserve-symlinks --preserve-symlinks-main tests/TestDatabase.mjs .cache/sql-fixture
```

The harness applies the migrations to a local database and prints the test results.

## Deployment

After deployment, score a sample and a fresh essay in the Space and from the local app. Check invalid input, language switching and session isolation.

Run `python Manage.py source-zip output/ielts-demo-source.zip` to package the source. Check the archive contents before publishing.

[Docs](../README.md)
