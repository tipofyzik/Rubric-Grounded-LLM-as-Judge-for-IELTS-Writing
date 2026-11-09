# IELTS Writing Evaluator

[Demo](https://tipofyzik-ielts-writing-evaluator.hf.space) · [Kaggle](code/offline_research/notebooks/README.md)

## Install and run locally

Use Python 3.11. From the repository root, install the local app.

```sh
python Manage.py install local
```

Copy [`.env.example`](.env.example) to `.env` and fill in your Hugging Face, Supabase and model release settings. You’ll need access to a compatible model release. See the [setup guide](docs/guides/04Installation.md) for the settings and the option to use local model files.

Download the model and reference files, then start the app.

```sh
python Manage.py setup --download-assets --warm-cache
python Manage.py local
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860).
