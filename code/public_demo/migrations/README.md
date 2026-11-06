# Migrations

Migration 004 creates the corpus schema/RLS/RPC; 005 adds raw text insights and updates RPC compatibility. A new database needs both in order and a separate corpus import. Existing production migrations are not rerun during deployment.

| File | Responsibility |
|---|---|
| [004_panda_corpus.sql](004_panda_corpus.sql) | Configuration, dependency or supporting source input. |
| [005_panda_text_insights.sql](005_panda_text_insights.sql) | Configuration, dependency or supporting source input. |

[Documentation catalogue](../../../docs/README.md)
