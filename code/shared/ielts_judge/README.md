# Ielts Judge

Own scoring contracts, prompts/parser, model registry, database reads and localized UI. ScoringRuntime.py wires loaded components; Scoring.py owns one bounded request.

| File | Responsibility |
|---|---|
| [__init__.py](__init__.py) | Research code for rubric-grounded IELTS writing evaluation. |
| [Access.py](Access.py) | Read only user runtime settings; source distributions never carry credentials. |
| [Criteria.py](Criteria.py) | Shared IELTS criterion definitions loaded from the project resource file. |
| [Database.py](Database.py) | Supabase HTTPS/RPC transport, idempotent corpus ingestion and typed retrieval. |
| [Localization.py](Localization.py) | Application UI text; language selection belongs to each browser session. |
| [ModelRegistry.py](ModelRegistry.py) | Approved judge profiles; no model loading or application startup on import. |
| [Paths.py](Paths.py) | Load project filesystem locations from the root paths.yaml file. |
| [Prompts.py](Prompts.py) | One Task 2 inference prompt shared by serving and research callers. |
| [Scores.py](Scores.py) | Public judge scores with exact fields and bounded format tolerance. |
| [Scoring.py](Scoring.py) | UI-independent scoring orchestration with one bounded prompt/judge/parser chain. |
| [ScoringRuntime.py](ScoringRuntime.py) | Explicit startup wiring for accepted AES/corpus releases; no loading on import. |
| [Ui.py](Ui.py) | Multilingual Gradio presentation of the UI-independent scoring service. |

[Documentation catalogue](../../../docs/README.md)
