# Shared code

Both apps use the same scoring engine and Gradio UI. The research tools reuse the scoring code for model comparisons.

- [`ielts_judge/`](ielts_judge/README.md) contains the scoring engine, reference search and Gradio UI.
- [`resources/`](resources/README.md) contains the rubric, demo examples and UI text.
- [`config.yaml`](config.yaml) sets the number of examples to retrieve. [`paths.yaml`](paths.yaml) sets the locations of datasets and model files.
