# Notes for reference essays

The PANDA adaptation asks a language model to explain scores already assigned to training essays. The prompt asks for evidence for each score and compares it with the neighboring bands. The app can then show the judge those notes when a new essay resembles one of the examples.

Start at [`ManageCorpus.py`](../ManageCorpus.py) to generate notes or prepare the reference collection. [`Generation.py`](Generation.py) writes the notes, and [`Corpus.py`](Corpus.py) pairs them with the essays and scores for search.

[Settings](../resources/README.md) · [Kaggle](../notebooks/README.md)
