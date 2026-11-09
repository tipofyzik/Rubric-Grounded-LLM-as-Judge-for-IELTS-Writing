# Supabase migrations

Run these scripts in order when setting up an empty Supabase database.

1. [`004_panda_corpus.sql`](004_panda_corpus.sql) creates the reference tables, access rules and search function.
2. [`005_panda_text_insights.sql`](005_panda_text_insights.sql) adds the original model notes to the database and search results.

Then import the reference essays. See [Deployment](../../../docs/guides/05Deployment.md).
