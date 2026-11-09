# Scoring engine

Start at [`Scoring.py`](Scoring.py) to follow a submission. It gets a first estimate, looks up similar essays, and calls the judge. [`Prompts.py`](Prompts.py) builds the prompt and [`Scores.py`](Scores.py) reads and validates the four scores.

| Part | What it does |
| --- | --- |
| [`aes/`](aes/README.md) | Estimates scores from the essay text |
| [`Database.py`](Database.py) and [`panda/`](panda/README.md) | Search for similar essays and read their scores and notes |
| [`llm/`](llm/README.md) | Loads and runs the judge model |
| [`Ui.py`](Ui.py) and [`Localization.py`](Localization.py) | Build the Gradio form and load its translations |
| [`data/`](data/README.md) | Reads the dataset and selects essays for experiments |

[Scoring flow](../../../docs/guides/02ScoringFlow.md)
