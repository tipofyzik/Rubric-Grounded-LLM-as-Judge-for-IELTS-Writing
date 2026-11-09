# Code


Start with the root manager, then read the shared service before platform adapters.

| Question | Source |
|---|---|
| Which command launches which component? | [Manage.py](../../Manage.py) |
| Where are runtime inputs restored? | [RuntimeAssets.py](../../RuntimeAssets.py) |
| Where is scoring coordinated? | [Scoring.py](../../code/shared/ielts_judge/Scoring.py) |
| What text does the judge receive? | [Prompts.py](../../code/shared/ielts_judge/Prompts.py) |
| What is a valid score response? | [Scores.py](../../code/shared/ielts_judge/Scores.py) |
| How are loaded models connected? | [ScoringRuntime.py](../../code/shared/ielts_judge/ScoringRuntime.py) |
| How does retrieval enforce corpus compatibility? | [Database.py](../../code/shared/ielts_judge/Database.py) |
| How is browser/session behavior implemented? | [Ui.py](../../code/shared/ielts_judge/Ui.py) |
| What differs on HF? | [Public App.py](../../code/public_demo/App.py), [CpuWorker.py](../../code/public_demo/CpuWorker.py) |
| What differs locally? | [Local Runtime.py](../../code/local_backup/Runtime.py) |

Class/function docstrings describe inputs and outputs. Annotations using `Any` mark dynamically shaped model, framework or record boundaries; they are not a claim of complete static type safety. Wire keys and historical artifact filenames keep their established spelling even though Python module filenames use PascalCase.


[Docs](../README.md)
