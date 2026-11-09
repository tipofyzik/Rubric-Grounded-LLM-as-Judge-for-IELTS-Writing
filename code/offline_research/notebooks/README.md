# Kaggle notebooks

Use these notebooks to train the essay model, write notes for reference essays or compare scoring methods on Kaggle. Each build command prepares a notebook and the input files it needs.

| Notebook | Build command | What you get |
| --- | --- | --- |
| [Train the essay model](AESTraining.ipynb) | `python Manage.py research bundle-aes` | A trained model and its scores on test essays |
| [Generate PANDA notes](PandaGeneration.ipynb) | `python Manage.py research bundle-panda` | Notes explaining scores on training essays |
| [Compare scoring methods](ModelEvaluation.ipynb) | `python Manage.py research bundle-evaluation` | Predictions, metrics and charts |

## Run on Kaggle

1. Run the build command from the repo root with the prepared dataset and [research dependencies](../requirements.txt) in place.
2. Import the generated notebook. Upload its matching ZIP to a private Kaggle Dataset and attach it with **Add Input**. For model comparison, include `ModelEvaluationBundleManifest.json` with the ZIP.
3. Enable a GPU and internet access for model runs. Follow the notebook cells and download the results from `/kaggle/working`.

The model comparison build also needs the trained model, reference essays and `docs/model_evaluation_runbook.md` from the full research checkout.

[ExportEvaluationResults.ipynb](ExportEvaluationResults.ipynb) exports a saved comparison.

[Kaggle Docs](https://www.kaggle.com/docs/notebooks) · [Evaluation report](../../../docs/model_evaluation_report.md)
