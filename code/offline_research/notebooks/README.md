# Kaggle

Use these notebooks to train the essay model, generate notes for reference essays or compare scoring methods on Kaggle. Run the builders from the repository root with the [research environment](../README.md) and a verified [prepared dataset](../data/README.md).

| Notebook template | Build command | Files from the build |
| --- | --- | --- |
| [Train the essay model](AESTraining.ipynb) | `python Manage.py research bundle-aes` | `AESTraining.ipynb`, `aes_training_bundle.zip` and `AESBundleManifest.json` |
| [Generate PANDA notes](PandaGeneration.ipynb) | `python Manage.py research bundle-panda` | `PandaGeneration.ipynb`, `panda_generation_bundle.zip` and `bundle_manifest.json` |
| [Compare scoring methods](ModelEvaluation.ipynb) | `python Manage.py research bundle-evaluation` | `ModelEvaluation.ipynb`, `model_evaluation_bundle.zip` and `ModelEvaluationBundleManifest.json` |

The model comparison also needs the AES and PANDA [release descriptors and their artifacts](../README.md#releases) from the same prepared training set. The builders print the output directory or the path to the generated ZIP.

## Run

1. Import the generated notebook from the output directory. Upload the ZIP from that same build to a private Kaggle Dataset and attach it with **Add Input**. For model comparison, upload `ModelEvaluationBundleManifest.json` alongside the ZIP.
2. Enable a GPU and internet access for model runs. The first cells verify that the input package matches the notebook. Follow the notebook cells and download the results from the printed path under `/kaggle/working`.
3. For model comparison, start with a small pilot before running the full pool. [Pilot, resume and export instructions](../evaluation/README.md)

## Legacy

Use `python Manage.py research bundle-legacy` to build `ielts_kaggle_bundle.zip` for [`KaggleNotebook.ipynb`](KaggleNotebook.ipynb). This workflow supports historical inference, fine-tuning and reporting.

[`ExportEvaluationResults.ipynb`](ExportEvaluationResults.ipynb) exports a saved model comparison.

[Kaggle Docs](https://www.kaggle.com/docs/notebooks) · [Evaluation report](../../../docs/model_evaluation_report.md)
