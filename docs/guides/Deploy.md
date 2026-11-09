# Deploy


The operator uses the existing Gradio/ZeroGPU Space, private artifact repository and Supabase database. Deployment does not create a Space, choose hardware or migrate the database.

1. Install the public operator profile and configure the explicit target and immutable artifact pin.
2. Run `python Manage.py public build`. Inspect the allowlisted source tree and `DEPLOYMENT.json`.
3. Run `python Manage.py public status` and confirm the target/revision.
4. Run `python Manage.py public deploy` when ready to update that Space. The previous revision is backed up and the update uses a parent-commit guard. `--configure-secrets` also sets the supported Space secrets.
5. Verify the new revision starts, serves the interface and scores a fresh input. Save the resulting deployment/acceptance evidence separately from historical releases.

The application restores hash-verified private AES/corpus artifacts at startup. Public source contains only code and safe configuration. No training, insights generation or database ingestion runs at startup. Runtime file routes remain blocked. A restart can require model initialization before the interface becomes ready.

For a new database, follow the [migration, corpus import and retrieval checks](../../code/public_demo/migrations/README.md). For routine deployments, use the existing compatible database and verify that its ready corpus matches the release used by the Space.

The local app uses the Space’s judge and must use the matching tokenizer. After updating the Space, verify that an essay can still be scored from the local app. If the deployment fails, check the saved build information and use the previous verified revision for rollback.


[Docs](../README.md)
