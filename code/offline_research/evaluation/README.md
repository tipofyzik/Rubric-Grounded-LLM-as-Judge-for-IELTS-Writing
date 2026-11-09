# Model evaluation

Run the essay scoring model and Qwen on the same test essays. The experiment varies the context given to Qwen and compares each set of predictions with the dataset scores.

- [`EvaluateModels.py`](../EvaluateModels.py) starts or resumes an experiment.
- [`Context.py`](Context.py) prepares the first score estimates and finds reference examples. [`Runtime.py`](Runtime.py) runs the judge and saves predictions.
- [`Report.py`](Report.py) computes the metrics. [`Plots.py`](Plots.py) draws the charts.

To rebuild a report from saved predictions, run this from the repository root and replace `path/to/run` with the saved run directory.

```sh
python Manage.py research evaluate --mode report --resume-run path/to/run
```

[Kaggle](../notebooks/README.md) · [Evaluation protocol](../../../docs/model_evaluation_protocol.md) · [Results](../../../docs/model_evaluation_report.md)
