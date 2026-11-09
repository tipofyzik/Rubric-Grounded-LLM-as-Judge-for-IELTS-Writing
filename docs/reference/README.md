# Source index

- [`Manage.py`](../../Manage.py) is the entry point for project commands. [`RuntimeAssets.py`](../../RuntimeAssets.py) restores the files used by the app.
- [`Scoring.py`](../../code/shared/ielts_judge/Scoring.py) runs a scoring request. [`Prompts.py`](../../code/shared/ielts_judge/Prompts.py) and [`Scores.py`](../../code/shared/ielts_judge/Scores.py) handle what goes into the model and what comes back.
- [`Ui.py`](../../code/shared/ielts_judge/Ui.py) builds the Gradio interface. [`Database.py`](../../code/shared/ielts_judge/Database.py) connects scoring to the reference database.
- [`App.py` for the Space](../../code/public_demo/App.py) starts the hosted demo. [`App.py` for local use](../../code/local_backup/App.py) starts the local app.
- [`PrepareData.py`](../../code/offline_research/PrepareData.py), [`TrainAes.py`](../../code/offline_research/TrainAes.py), [`ManageCorpus.py`](../../code/offline_research/ManageCorpus.py) and [`EvaluateModels.py`](../../code/offline_research/EvaluateModels.py) are the research entry points.

[Project layout](../../code/README.md)
