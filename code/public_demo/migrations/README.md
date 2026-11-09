# Set up reference search

These scripts prepare a new Supabase database to store scored training essays and find similar ones.

1. [`004_panda_corpus.sql`](004_panda_corpus.sql) creates the tables and adds the search function.
2. [`005_panda_text_insights.sql`](005_panda_text_insights.sql) adds the notes that explain each example’s scores.

Run the scripts in order, then import the reference essays. [Deployment](../../../docs/guides/05Deployment.md)
