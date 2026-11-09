# Public demo operations

Use `python Manage.py install public` for operator dependencies. This installs CPU tooling locally; the Space runtime uses its separate pinned `requirements.txt` and the SDK metadata in `README.md`.

Configure your own HF Space, artifact repository and Supabase in `.env`. For a new database, follow the [migration, import and retrieval checks](migrations/README.md). Use the Supabase publishable key for the Space and keep the secret key in the import environment.

If you need to publish your verified local model and corpus to a private Hugging Face repository, run `python Manage.py public artifacts` first. For an existing release, provide its immutable artifact settings as described in the [installation guide](../../docs/guides/Install.md).

```powershell
python Manage.py public build
python Manage.py public status
python Manage.py public deploy --configure-secrets
```

`build` is local and prepares `results/demo/build/public-space`. The generated tree is an independent Gradio Space: root `App.py`, worker/runtime files, minimal `code/shared` and `DEPLOYMENT.json`. Do not launch this source-directory `App.py` as if the deployment tree already existed.

`artifacts` and `deploy` are real uploads to the repositories you configure. The scripts do not create/upgrade Space hardware. Inspect the generated output before deployment. `IELTS_MODEL_KEY` chooses one enabled profile at startup; local backup must use the same model/tokenizer pin. Switching weights inside a request is not supported.

The CPU worker remains separate from the ZeroGPU worker. Queue, deadlines, prompts and validation are preserved. No training or database migration occurs during application startup.
