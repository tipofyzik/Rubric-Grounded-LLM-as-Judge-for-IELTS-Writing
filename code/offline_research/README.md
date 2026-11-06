# Offline Research

Preserve data preparation, AES training, PANDA generation/corpus tooling and model evaluation. Research is deferred for this release. Explicit Manage.py research commands own these operations; demo startup never invokes them. Data paths are configured in shared/paths.yaml under the separate local data directory.

| File | Responsibility |
|---|---|
| [__init__.py](__init__.py) | IELTS offline_research component. |
| [config.yaml](config.yaml) | Configuration, dependency or supporting source input. |
| [EvaluateModels.py](EvaluateModels.py) | Run standalone model evaluation; never call or deploy the hosted service. |
| [ManageCorpus.py](ManageCorpus.py) | Offline PANDA generation, corpus building and Supabase RPC verification. |
| [PrepareData.py](PrepareData.py) | Run explicitly enabled research stages against a verified local Parquet release. |
| [requirements-evaluation.txt](requirements-evaluation.txt) | Configuration, dependency or supporting source input. |
| [requirements.txt](requirements.txt) | Configuration, dependency or supporting source input. |
| [TrainAes.py](TrainAes.py) | Train, audit or predict with an essay-only AES bundle (no judge/retrieval). |

[Documentation catalogue](../../docs/README.md)
