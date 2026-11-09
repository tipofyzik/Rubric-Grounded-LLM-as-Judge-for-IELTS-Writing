# Kaggle notebooks

Build the notebook and its input files together before importing it into Kaggle.

| Task | Build command | Output from the notebook |
| --- | --- | --- |
| [Train the essay scorer](AESTraining.ipynb) | `python Manage.py research bundle-aes` | Model weights and scores on the test set |
| [Generate PANDA notes](PandaGeneration.ipynb) | `python Manage.py research bundle-panda` | Notes explaining the scores of training essays |
| [Compare scoring methods](ModelEvaluation.ipynb) | `python Manage.py research bundle-evaluation` | Predictions, metrics and charts |

## Run on Kaggle

1. Run the build command from the repository root with the [research dependencies](../requirements.txt) and the prepared dataset in place. Model comparison also needs the trained essay scorer, the reference set and `docs/model_evaluation_runbook.md` from the full research checkout.
2. Import the generated notebook into Kaggle. Upload its matching ZIP to a private Kaggle Dataset and attach the dataset with **Add Input**. For model comparison, upload `ModelEvaluationBundleManifest.json` alongside the ZIP.
3. Enable a GPU and internet access for model runs. Follow the notebook cells and download the results from `/kaggle/working`.

[ExportEvaluationResults.ipynb](ExportEvaluationResults.ipynb) exports a saved evaluation. The [legacy notebook](KaggleNotebook.ipynb) covers fine-tuning, inference and reporting. Build its input ZIP with `python Manage.py research bundle-legacy`.

[Kaggle Docs](https://www.kaggle.com/docs/notebooks) · [Evaluation report](../../../docs/model_evaluation_report.md)
