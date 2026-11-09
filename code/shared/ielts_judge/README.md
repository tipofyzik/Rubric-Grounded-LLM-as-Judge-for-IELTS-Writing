# Scoring engine

Every submission goes through this code. It makes a first estimate, finds scored examples, puts together the instructions for the judge, and checks the reply. Start at [`Scoring.py`](Scoring.py) to follow the whole process.

- [`aes/`](aes/README.md) makes the first estimate from the essay text.
- [`Database.py`](Database.py) finds similar essays. [`panda/`](panda/README.md) handles the example scores and notes.
- [`Prompts.py`](Prompts.py) puts the rubric and examples into the prompt. [`llm/`](llm/README.md) runs the judge, and [`Scores.py`](Scores.py) checks the scores it returns.
- [`Ui.py`](Ui.py) builds the form, and [`Localization.py`](Localization.py) loads the translations.
- [`data/`](data/README.md) helps the research tools work with the same essay dataset.

[Scoring flow](../../../docs/guides/02ScoringFlow.md)
