# PANDA

Scored training essays give the judge examples to work from. This code reads and checks the collection the app searches when it needs a similar essay.

[`Artifacts.py`](Artifacts.py) handles saved completions and source scores. [`Corpus.py`](Corpus.py) checks the reference manifest and records, and provides the encoder for the search vector. The prompt uses the retrieved scores and notes. [`Prompts.py`](../Prompts.py) defines the fields passed to the judge.

[Generate PANDA notes](../../../offline_research/panda/README.md)
