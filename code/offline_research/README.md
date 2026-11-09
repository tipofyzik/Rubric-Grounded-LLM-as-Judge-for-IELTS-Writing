# Research

Prepare scored essays, train the automated essay scorer (AES), build the PANDA reference corpus and compare prompt variants. The datasets and trained models are separate inputs.

Run `python Manage.py install research` from the repository root with Python 3.11. Use CUDA or Kaggle for PANDA generation and the default Qwen benchmark.

## Workflow

| Task | Input | Output |
| --- | --- | --- |
| [Prepare data](data/README.md) | Chillies and Writing9 | Train and test data |
| [Train AES](aes/README.md) | Prepared data | Essay scorer and test metrics |
| [Build PANDA](panda/README.md) | Prepared data and generated notes | Reference corpus |
| [Evaluate](evaluation/README.md) | Matching data, AES and PANDA | Predictions and metrics |

[Kaggle](notebooks/README.md) · [Legacy LLM tools](llm/README.md) · [Release layout](../../docs/guides/Install.md#release-layout) · [Results](../../docs/model_evaluation_report.md)
