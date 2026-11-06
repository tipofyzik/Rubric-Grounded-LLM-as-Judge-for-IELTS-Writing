# Reference corpus: схема PostgreSQL и RPC

Для новой Supabase БД порядок: `004_panda_corpus.sql`, затем
`005_panda_text_insights.sql`, затем отдельный импорт принятого corpus через
`code/panda.py ingest`. В действующей БД проекта обе миграции уже применены;
обычный запуск локального/облачного приложения не исполняет SQL-миграции.

**004: основа хранения и доступа**

Файл создаёт `extensions.vector`, `extensions.pgcrypto` и схему `ielts`.

| Таблица | Что хранит |
|---|---|
| `corpus_releases` | Версию/manifest corpus, `draft`/`ready` |
| `reference_essays` | Train-only ID/hash, Task/Essay, source/provenance, 768-dimensional BGE vector и truncation metadata |
| `reference_criteria` | Четыре исходных integer scores 0–9 и structured insight каждого критерия |
| `rubric_passages` | Рубрику и её provenance из generator manifest |

RLS разрешает anon/authenticated чтение готовых releases. Права записи и
импортные RPC получает `service_role`. Functions используют SECURITY INVOKER,
то есть права вызывающего, и ограничения таблиц/RLS продолжают действовать.

| RPC | Назначение | Кто вызывает |
|---|---|---|
| `panda_begin` | Проверить manifest, зарегистрировать release и рубрику | Offline importer |
| `panda_import` | Загрузить batches references с проверкой ID/hash/labels и повторного импорта | Offline importer |
| `panda_finalize` | Проверить полный count/digest и сделать release `ready` | Offline importer |
| `panda_match` | Проверить corpus/encoder profile, найти похожие essays, исключить query ID/hash | Runtime retrieval |

`rubric_passages` служит provenance corpus. Prompt приложения получает
рубрику из локального `code/resources/ielts_criteria.yaml`; runtime её из БД
не запрашивает.

**005: raw text corpus v2**

Файл добавляет `reference_essays.insights_text`, `generation_status`,
`generation_error`. Это позволяет хранить исходный completion целиком, в том
числе failed/partial, без исправления текста и source scores.

Переопределяются `panda_begin`, `panda_import` и `panda_match`: они принимают v1/v2,
проверяют подходящий формат и возвращают raw text для v2. Structured v1 остаётся
читаемым. Таблицы, RLS/grants и `panda_finalize` из 004 остаются необходимыми.
Поэтому 005 не заменяет всю миграцию 004 и не работает отдельно на пустой БД.

Ни одна миграция не создаёт реальные 8 236 references: это делает отдельный
импорт готового corpus. Для обычного scoring нужен только publishable key;
import/admin secret приложению не передаётся. Пользовательские эссе не становятся
reference rows при оценивании.

Проверка схемы: `code/tests/TestDatabase.mjs` запускает настоящий PostgreSQL
через PGlite/pgvector на synthetic fixture. Она проверяет migrations/reapply,
RLS, import/reimport, raw-text roundtrip, cosine order и guards; production БД
при этом не затрагивается. Команды — в [PANDA runbook](../docs/panda_runbook.md).
