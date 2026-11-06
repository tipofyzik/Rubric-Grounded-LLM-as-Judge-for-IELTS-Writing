# Llm

Load pinned judge profiles, count rendered chat tokens, generate under the deadline and enforce the shared completion/EOS behavior.

| File | Responsibility |
|---|---|
| [__init__.py](__init__.py) | Language model fine-tuning and response generation. |
| [Generation.py](Generation.py) | Single-response HF generation shared by research and serving callers. |
| [JsonEosBlocker.py](JsonEosBlocker.py) | Prevent EOS while an IELTS score object is incomplete. |
| [Judge.py](Judge.py) | Public single-submission judge call, independent of UI and research writers. |

[Documentation catalogue](../../../../docs/README.md)
