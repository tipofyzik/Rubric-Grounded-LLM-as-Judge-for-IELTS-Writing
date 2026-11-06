# Notebooks

Store research notebook templates and legacy bootstrap tooling. Build a new matching ZIP/manifest/notebook after source changes; a new ZIP must not be paired with an old pinned notebook. No new GPU runs are part of the engineering release.

| File | Responsibility |
|---|---|
| [__init__.py](__init__.py) | Offline research notebooks workflows. |
| [AESTraining.ipynb](AESTraining.ipynb) | Configuration, dependency or supporting source input. |
| [BuildBundle.py](BuildBundle.py) | Build the Kaggle runtime ZIP from shared code and a verified dataset release. |
| [ExportEvaluationResults.ipynb](ExportEvaluationResults.ipynb) | Configuration, dependency or supporting source input. |
| [KaggleNotebook.ipynb](KaggleNotebook.ipynb) | Configuration, dependency or supporting source input. |
| [ModelEvaluation.ipynb](ModelEvaluation.ipynb) | Configuration, dependency or supporting source input. |
| [ModelEvaluationConfig.yaml](ModelEvaluationConfig.yaml) | Configuration, dependency or supporting source input. |
| [ModelEvaluationModels.yaml](ModelEvaluationModels.yaml) | Configuration, dependency or supporting source input. |
| [PandaGeneration.ipynb](PandaGeneration.ipynb) | Configuration, dependency or supporting source input. |
| [RunChecks.py](RunChecks.py) | Prepare a small Kaggle fixture, rebuild its ZIP and run local notebook checks. |
| [RunConfig.yaml](RunConfig.yaml) | Configuration, dependency or supporting source input. |

[Documentation catalogue](../../../docs/README.md)
