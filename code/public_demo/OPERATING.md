# Public demo operations

Use `python Manage.py install public` for operator dependencies. This installs CPU tooling locally; the Space runtime uses its separate pinned `requirements.txt` and the SDK metadata in `README.md`.

Configure your own HF Space, artifact repository and Supabase in `.env`. Apply migrations 004 then 005 to your new database and import the verified corpus using the research CLI. Never expose the import secret to the public runtime.

```powershell
python Manage.py public artifacts
python Manage.py public build
python Manage.py public deploy --configure-secrets
python Manage.py public status
```

`build` is local and prepares `results/demo/build/public-space`. The generated tree is an independent Gradio Space: root `App.py`, worker/runtime files, minimal `code/shared` and `DEPLOYMENT.json`. Do not launch this source-directory `App.py` as if the deployment tree already existed.

`artifacts` and `deploy` are real uploads to the repositories you configure. The scripts do not create/upgrade Space hardware. Inspect the generated output before deployment. `IELTS_MODEL_KEY` chooses one enabled profile at startup; local backup must use the same model/tokenizer pin. Switching weights inside a request is not supported.

The CPU worker remains separate from the ZeroGPU worker. Queue, deadlines, prompts and validation are preserved. No training or database migration occurs during application startup.
