# Kaggle

Build the Kaggle inputs from the repository root with the [research environment](../README.md) and a [prepared dataset](../data/README.md).

| Notebook | Build command |
| --- | --- |
| [Train AES](AESTraining.ipynb) | `python Manage.py research bundle-aes` |
| [Generate PANDA notes](PandaGeneration.ipynb) | `python Manage.py research bundle-panda` |
| [Evaluate](ModelEvaluation.ipynb) | `python Manage.py research bundle-evaluation` |

Evaluation also needs matching AES and PANDA releases. [Release layout](../../../docs/guides/Install.md#release-layout)

## Run

1. Import the generated notebook. Upload the ZIP from the same build to a private Kaggle Dataset and attach it with **Add Input**. For evaluation, include `ModelEvaluationBundleManifest.json` alongside the ZIP.
2. Enable a GPU and internet access, then run the notebook and download its results ZIP.

[Evaluation and resume](../evaluation/README.md)

## Legacy

Run `python Manage.py research bundle-legacy` for [`KaggleNotebook.ipynb`](KaggleNotebook.ipynb).

[`ExportEvaluationResults.ipynb`](ExportEvaluationResults.ipynb) exports a saved model comparison.

[Kaggle Docs](https://www.kaggle.com/docs/notebooks) · [Evaluation report](../../../docs/model_evaluation_report.md)
