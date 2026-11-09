# Migrations

Apply these scripts to a new Supabase database in order.

1. [`004_panda_corpus.sql`](004_panda_corpus.sql) creates the tables and adds the search function.
2. [`005_panda_text_insights.sql`](005_panda_text_insights.sql) adds the notes that explain each example’s scores.

From the repository root, use the [research environment](../../offline_research/README.md) and a [built corpus](../../offline_research/panda/README.md#build). Set `SUPABASE_URL` and `SUPABASE_SECRET_KEY` for the import.

```sh
python Manage.py research panda ingest --corpus results/panda/corpus
```

When the import reports `ready`, set `SUPABASE_PUBLISHABLE_KEY` and check retrieval.

```sh
python Manage.py research panda verify-rpc --corpus results/panda/corpus --output results/panda/rpc-verification.json
```

A successful check reports `passed`. [Deployment](../../../docs/guides/Deploy.md)
