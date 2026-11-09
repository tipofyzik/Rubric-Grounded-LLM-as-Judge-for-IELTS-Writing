# Research

The research tools prepare scored essays, train the automated essay scorer (AES), build the PANDA reference corpus and compare prompt variants. A source checkout provides the code and configs. The original datasets, prepared Parquet release, trained weights, reference corpus and saved runs are separate inputs.

Use Python 3.11 and run `python Manage.py install research` from the repository root for the local CPU tools. Use a CUDA environment for PANDA generation and the default judge benchmark. The Kaggle notebooks set up their own dependencies around the available CUDA PyTorch.

## Workflow

Run these commands from the repository root.

| Task | Command | Input | Output |
| --- | --- | --- | --- |
| [Prepare data](data/README.md) | `python Manage.py research data --prepare-only` | Source CSV/JSONL with transformation enabled, or a verified prepared release | `ielts-v3` Parquet and manifest feed AES, PANDA and evaluation. The command also records the experiment selection. |
| [Train AES](aes/README.md) | `python Manage.py research aes train --output results/aes/runs/my-run` | Prepared dataset and AES settings | `aes_bundle/`, predictions and metrics. |
| [Generate notes](panda/README.md) | `python Manage.py research panda generate --output results/panda/insights` | Prepared dataset, generator settings and CUDA | Generation records for all training essays. |
| [Build corpus](panda/README.md#build) | `python Manage.py research panda build --insights results/panda/insights --output results/panda/corpus` | Prepared dataset, generation records and embedding model | Corpus for evaluation and [Supabase import](../public_demo/migrations/README.md). |
| [Evaluate](evaluation/README.md) | `python Manage.py research evaluate --mode all` | Matching prepared data, AES and corpus releases, and configured models | Predictions, coverage, metrics and charts. |

## Releases

Keep the validated model bundle and corpus under the project root and create the corresponding release descriptors. The table lists the fields used to load the assets.

| Descriptor | Required fields |
| --- | --- |
| `results/aes/release.json` | Set `paths_relative_to` to `this_file_parent`. Set `bundle_path` relative to the descriptor, for example `runs/my-run/aes_bundle`. Use the bundle manifest hash from `aes_report.json` for `bundle_manifest_sha256`. |
| `results/panda/release.json` | Set `corpus_path` relative to the project root, for example `results/panda/corpus`. Copy `corpus_id` and the integer `count` from the corpus manifest. |

For the model benchmark, the dataset and both releases must identify the same prepared training set. For the full app, Supabase must contain the same corpus in state `ready`. Keep the descriptor paths and the files they reference together when packaging a release for [local setup](../../docs/guides/Install.md).

[Kaggle notebooks](notebooks/README.md) · [Historical LLM workflows](llm/README.md) · [Published results](../../docs/model_evaluation_report.md)
