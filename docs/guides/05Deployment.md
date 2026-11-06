# Operating the existing HF demo


The operator uses the existing Gradio/ZeroGPU Space, private artifact repository and Supabase database. Deployment does not create a Space, choose hardware or migrate the database.

1. Install the public operator profile and configure the explicit target and immutable artifact pin.
2. Run `python Manage.py public build`. Inspect the allowlisted source tree and `DEPLOYMENT.json`.
3. Run `python Manage.py public status` and confirm the target/revision.
4. Run `python Manage.py public deploy` when ready to update that Space. The previous revision is backed up and the update uses a parent-commit guard. `--configure-secrets` also sets the supported Space secrets.
5. Verify the new revision starts, serves the interface and scores a fresh input. Save the resulting deployment/acceptance evidence separately from historical releases.

The application restores hash-verified private AES/corpus artifacts at startup. Public source contains only code and safe configuration. No training, insights generation or database ingestion runs at startup. Runtime file routes remain blocked. A restart can require model initialization before the interface becomes ready.

For a new database only: apply migration 004 then 005, then separately import the compatible corpus using the offline import secret. Existing accepted Supabase installations already have these migrations; routine deployment does not repeat them.

The local adapter depends on the preserved `stage5_judge` endpoint and matching tokenizer profile. Verify local fallback after changing the public runtime. A deployment failure should be diagnosed against saved source/revision evidence; use the saved previous revision as the rollback source without changing data or artifact versions blindly.


[Documentation catalogue](../README.md)
