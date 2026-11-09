# Scoring

The app passes the question and essay to [`Scoring.py`](Scoring.py). The engine combines an essay-only estimate, scores and notes from a retrieved training example, and the rubric in a prompt for Qwen. It validates the reply and returns four integer scores from 0 to 9, or a structured error.

- [`aes/`](aes/README.md) makes the first estimate from the essay text.
- [`Database.py`](Database.py) searches for a similar essay using a BGE vector and excludes matching essay fingerprints. [`panda/`](panda/README.md) handles the corpus, encoder, source scores and notes.
- [`Prompts.py`](Prompts.py) puts the rubric and examples into the prompt. [`llm/`](llm/README.md) runs the judge, and [`Scores.py`](Scores.py) checks the scores it returns.
- [`Ui.py`](Ui.py) builds the form, and [`Localization.py`](Localization.py) loads the translations.
- [`data/`](data/README.md) helps the research tools work with the same essay dataset.

[Scoring flow](../../../docs/guides/Scoring.md)
