# Resources

Edit these files to change what people see in the app or what the judge uses to score an essay.

- [`demo_examples.json`](demo_examples.json) contains the sample questions and essays loaded by the [interface](../ielts_judge/Ui.py).
- [`ielts_criteria.yaml`](ielts_criteria.yaml) contains the rubric read by [`Criteria.py`](../ielts_judge/Criteria.py).
- [`model_registry.yaml`](model_registry.yaml) pins the judge and tokenizer revision and sets its token budgets. The hosted and local profiles must match. [Registry loader](../ielts_judge/ModelRegistry.py)
- [`locales/`](locales/README.md) contains the labels and messages shown in the app.
