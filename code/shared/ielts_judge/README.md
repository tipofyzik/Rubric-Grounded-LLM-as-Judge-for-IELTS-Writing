# Scoring

[`Scoring.py`](Scoring.py) combines the rubric, preliminary scores and reference notes in a prompt for Qwen. It checks the reply and returns four criterion scores.

- [`aes/`](aes/README.md) makes the first estimate from the essay text.
- [`Database.py`](Database.py) finds a similar essay in the [PANDA corpus](panda/README.md).
- [`Prompts.py`](Prompts.py) puts the rubric and examples into the prompt. [`llm/`](llm/README.md) runs the judge, and [`Scores.py`](Scores.py) checks the scores it returns.
- [`Ui.py`](Ui.py) builds the form, and [`Localization.py`](Localization.py) loads the translations.
- [`data/`](data/README.md) helps the research tools work with the same essay dataset.

[Scoring flow](../../../docs/guides/Scoring.md)
