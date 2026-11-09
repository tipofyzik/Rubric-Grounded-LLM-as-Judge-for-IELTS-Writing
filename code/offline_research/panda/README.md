# PANDA

The PANDA adaptation asks a language model to explain scores already assigned to training essays. The prompt asks for evidence for each score and compares it with the neighboring bands. The app can then show the judge those notes when a new essay resembles one of the examples.

## Generate

Use a verified [prepared dataset](../data/README.md) and the [generator settings](../resources/README.md). Run from the repository root in a CUDA environment or use the [Kaggle notebook](../notebooks/README.md). A small pilot can use `--limit` to cap new attempts. Rerun with the same output directory and settings to continue generation.

```sh
python Manage.py research panda generate --output results/panda/insights
```

The run contains `generation_manifest.json` and per-essay generation records. The separate `compare` command tests two prompt variants on a fixed subset.

## Build

The corpus builder requires a generation record with a saved completion for every training essay, including failed generations. It preserves those raw completions. A prompt-comparison run or a pilot that covers only part of train cannot serve as the corpus input.

```sh
python Manage.py research panda build --insights results/panda/insights --output results/panda/corpus
python Manage.py research panda inspect --corpus results/panda/corpus
```

Build creates `manifest.json` and `records.jsonl` in a new directory. Inspect prints the verified manifest, including the corpus ID and count. Use those values in the [release descriptor](../README.md#releases), then follow the [Supabase import guide](../../public_demo/migrations/README.md) for hosted or local app retrieval.

Start at [`ManageCorpus.py`](../ManageCorpus.py) to generate notes or prepare the reference collection. [`Generation.py`](Generation.py) writes the notes, and [`Corpus.py`](Corpus.py) pairs them with the essays and scores for search.
