# Tools

Build matching research or historical probe source bundles. Packaging does not execute training or judge inference. Each generated notebook pins the matching source/data manifest.

| File | Responsibility |
|---|---|
| [__init__.py](__init__.py) | Standalone visual tools used with the research artifacts. |
| [BuildAesKaggle.py](BuildAesKaggle.py) | Build a matching AES notebook and verified prepared-data snapshot, without training. |
| [BuildEvaluationKaggle.py](BuildEvaluationKaggle.py) | Create a NEW verified evaluation bundle and matching notebook; no uploads. |
| [BuildPandaKaggle.py](BuildPandaKaggle.py) | Package PANDA generation code, verified data and a hash-pinned Kaggle notebook. |
| [BuildStage1Space.py](BuildStage1Space.py) | Build an allowlisted Space probe using the current shared Stage 1 source. |
| [PromptImage.py](PromptImage.py) | Python package marker. |

[Documentation catalogue](../../../docs/README.md)
