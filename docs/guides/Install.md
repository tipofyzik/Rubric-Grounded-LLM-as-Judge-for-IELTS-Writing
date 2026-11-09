# Install

## Install dependencies

Use Python 3.11. Run `python Manage.py install local` from the repository root to create the project's CPU environment. Use `install public` for deployment tools. The [Kaggle notebooks](../../code/offline_research/notebooks/README.md) manage their own CUDA environments.

## Configure access

Create `.env` from `.env.example` and fill in your values. Process variables take precedence. Pass `--env-file PATH` before the command to select another settings file.

| Setting | Purpose |
|---|---|
| `HF_SPACE_ID` | Explicit `owner/name` of the existing Space |
| `HF_TOKEN` | Authorized private-artifact download, remote judge or operator deployment |
| `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY` | Runtime read RPC |
| `SUPABASE_SECRET_KEY` | Offline import only; not passed to the scoring runtime |
| `IELTS_MODEL_KEY` | Enabled pinned model/tokenizer profile |
| `HF_ARTIFACT_REPO`, `HF_ARTIFACT_REVISION`, `HF_ARTIFACT_MANIFEST_SHA256` | Accessible immutable AES/corpus release |

## Restore assets

Use a compatible release you can access. The author's Hugging Face artifact repository is private. To download a release, run `python Manage.py setup --download-assets --warm-cache`. For a release on disk, run `python Manage.py setup --artifact-source "PATH_TO_RELEASE" --warm-cache` and point to the extracted root. Setup verifies and restores the release under `results/aes/` and `results/panda/`.

To import releases already present in an older checkout, run `python Manage.py setup --artifact-source . --offline`. This uses the files on disk. A fresh cache needs the pinned BGE model and Qwen tokenizer. The Qwen weights run in the Space.

## Run the app

Run `python Manage.py local --check`, then `python Manage.py local`. Score one sample essay to check retrieval and inference. [CLI](../../code/local_backup/README.md) · [Docs](../README.md)

## Release layout

Keep the trained bundle and corpus under the project root. For a new release, create these two files.

| File | Required fields |
| --- | --- |
| `results/aes/release.json` | Set `paths_relative_to` to `this_file_parent` and `bundle_path` to the bundle path relative to the descriptor. Copy `bundle_manifest_sha256` from `aes_report.json`. |
| `results/panda/release.json` | Set `corpus_path` relative to the project root. Copy `corpus_id` and the integer `count` from the corpus manifest. |

The benchmark requires the prepared dataset and both releases to agree on the training set. App deployment requires the matching corpus in Supabase in state `ready`. Package the release files together with the bundle and corpus they reference. [Deploy](Deploy.md)
