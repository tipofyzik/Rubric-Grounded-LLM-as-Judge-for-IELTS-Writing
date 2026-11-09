# Shared code

The hosted and local apps use the same scoring rules and interface. The research tools use the same scoring code when comparing models.

- [`ielts_judge/`](ielts_judge/README.md) takes an essay from submission to scores.
- [`resources/`](resources/README.md) stores the rubric, examples and interface text.
- [`config.yaml`](config.yaml) sets how many examples the app looks up. [`paths.yaml`](paths.yaml) tells the project where to find data and model files.
