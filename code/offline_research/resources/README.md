# PANDA settings

These files tell the research tools how to generate notes for scored essays and how to compare different ways of asking for those notes.

- [`panda_generation.yaml`](panda_generation.yaml) configures the generator, prompt variant and embedding model.
- [`panda_comparison.json`](panda_comparison.json) records selected essay IDs and hashes for a specific prepared release. [`Comparison.py`](../panda/Comparison.py) selects the two prompt variants, `simple_json` and `legacy_text`.

[Reference notes](../panda/README.md)
