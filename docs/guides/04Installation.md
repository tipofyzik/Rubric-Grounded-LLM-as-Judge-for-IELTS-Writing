# Settings, dependencies and compatible artifacts


Use Python 3.11 for operator/local commands. `python Manage.py install local` creates an isolated CPU environment. Choose `install public` for deployment tools. Research dependencies are separate; the research CPU installer must not replace CUDA Torch inside a GPU notebook.

Create a new `.env` from `.env.example`, then fill your own values. Existing files must not be overwritten. The manager reads a supported whitelist in memory; process variables override file values. `--env-file PATH` selects another settings file.

| Setting | Purpose |
|---|---|
| `HF_SPACE_ID` | Explicit `owner/name` of the existing Space |
| `HF_TOKEN` | Authorized private-artifact download, remote judge or operator deployment |
| `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY` | Runtime read RPC |
| `SUPABASE_SECRET_KEY` | Offline import only; not passed to the scoring runtime |
| `IELTS_MODEL_KEY` | Enabled pinned model/tokenizer profile |
| `HF_ARTIFACT_REPO`, `HF_ARTIFACT_REVISION`, `HF_ARTIFACT_MANIFEST_SHA256` | Accessible immutable AES/corpus release |

The author's artifact repository is private. A source checkout does not grant access. An operator must supply an accessible compatible release or its verified local directory. Research datasets are separate inputs; their expected filenames and locations are configured in [shared paths.yaml](../../code/shared/paths.yaml). Acquisition and redistribution of those datasets are outside the demo installation.

Run `python Manage.py setup --download-assets --warm-cache` to download the model and reference files. If you have a compatible release on disk, use `python Manage.py setup --artifact-source PATH_TO_RELEASE --warm-cache`. Point `PATH_TO_RELEASE` to the root of the extracted release and keep its model files, reference data and manifests in their original layout. Setup checks the files before copying them.

An empty cache requires the pinned BGE model and judge tokenizer. Qwen weights belong to the hosted runtime. Then run `python Manage.py local --check` and `python Manage.py local`. See [local operation](../../code/local_backup/README.md).


[Documentation catalogue](../README.md)
