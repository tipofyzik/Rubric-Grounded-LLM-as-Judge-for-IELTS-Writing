# Migrations

These scripts prepare a new Supabase database to store scored training essays and find similar ones.

1. [`004_panda_corpus.sql`](004_panda_corpus.sql) creates the tables and adds the search function.
2. [`005_panda_text_insights.sql`](005_panda_text_insights.sql) adds the notes that explain each example’s scores.

For a new Supabase database, apply the scripts in order. Migration 004 requires the `vector` and `pgcrypto` extensions in the `extensions` schema and creates them there when absent.

From the repository root, use the [research environment](../../offline_research/README.md) and a [verified corpus directory](../../offline_research/panda/README.md#build) containing `manifest.json` and `records.jsonl`. Set `SUPABASE_URL` and `SUPABASE_SECRET_KEY` in the environment or `.env` for the import.

```sh
python Manage.py research panda ingest --corpus results/panda/corpus
```

Check that the returned `state` is `ready` and the corpus ID and count match the manifest. Then set `SUPABASE_PUBLISHABLE_KEY` and check retrieval with the same read access as the app.

```sh
python Manage.py research panda verify-rpc --corpus results/panda/corpus --output results/panda/rpc-verification.json
```

A successful check writes `"status": "passed"` and the reference IDs found for the verification queries. Keep the secret key for import and configure the app with the publishable key. [Deployment](../../../docs/guides/Deploy.md)
