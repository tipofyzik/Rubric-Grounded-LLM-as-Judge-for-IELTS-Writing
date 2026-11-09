# Generate PANDA notes

PANDA asks a language model to explain the existing scores on training essays. When the app scores a new essay, it can give the judge the notes from a similar one.

Start at [`ManageCorpus.py`](../ManageCorpus.py) to generate the notes or build the reference set.

- [`InsightGenerator.py`](InsightGenerator.py) builds prompts for the notes. [`Generation.py`](Generation.py) runs them in batches.
- [`Corpus.py`](Corpus.py) joins the saved notes to the scored essays and prepares the reference set for search.
- [`RetrievalEvaluator.py`](RetrievalEvaluator.py) tests scoring with retrieved examples.

[Generation settings](../resources/README.md) · [Kaggle](../notebooks/README.md)
