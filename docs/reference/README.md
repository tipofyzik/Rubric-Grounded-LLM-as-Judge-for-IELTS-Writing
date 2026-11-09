# Source index

Use this list to jump to the main files for running the app, scoring an essay or comparing models.

- [`Manage.py`](../../Manage.py) is where project commands start. [`RuntimeAssets.py`](../../RuntimeAssets.py) gets the files the app needs.
- [`Scoring.py`](../../code/shared/ielts_judge/Scoring.py) handles an essay from submission to scores. [`Prompts.py`](../../code/shared/ielts_judge/Prompts.py) builds the prompt, and [`Scores.py`](../../code/shared/ielts_judge/Scores.py) checks the reply.
- [`Ui.py`](../../code/shared/ielts_judge/Ui.py) builds the form. [`Database.py`](../../code/shared/ielts_judge/Database.py) finds similar scored essays.
- [Hosted app](../../code/public_demo/App.py) and [local app](../../code/local_backup/App.py) start the two versions of the interface.
- [Research](../../code/offline_research/README.md) links to the tools for preparing data, training models and comparing scores.
