# PANDA

PANDA adds notes to scored training essays. The judge uses scores and notes from a similar essay as context.

## Generate

Use a [prepared dataset](../data/README.md) and run from the repository root with CUDA. You can also [run on Kaggle](../notebooks/README.md). Use `--limit` for a pilot. Rerun with the same output directory and settings to resume.

```sh
python Manage.py research panda generate --output results/panda/insights
```

## Build

Use the generation output for the full training split, including failed records.

```sh
python Manage.py research panda build --insights results/panda/insights --output results/panda/corpus
python Manage.py research panda inspect --corpus results/panda/corpus
```

The first command builds the corpus. The second checks it and prints the corpus ID and size.

[`ManageCorpus.py`](../ManageCorpus.py) · [Settings](../resources/README.md) · [Release layout](../../../docs/guides/Install.md#release-layout) · [Supabase import](../../public_demo/migrations/README.md)
