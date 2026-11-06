# Evaluation

Own separate prepare/infer/report/export model experiments. Report mode can redraw saved predictions without inference; default all mode may invoke models. New experiments are deferred.

| File | Responsibility |
|---|---|
| [__init__.py](__init__.py) | Offline research evaluation workflows. |
| [Config.py](Config.py) | Explicit experiment and report options; plotting never changes inference. |
| [Context.py](Context.py) | Prepare real AES/BGE context once, using the accepted train corpus offline. |
| [LLMAggregator.py](LLMAggregator.py) | Python package marker. |
| [LLMEvaluator.py](LLMEvaluator.py) | CSV entry point for the shared model evaluation tables and plots. |
| [Metrics.py](Metrics.py) | Integer-band metrics shared by AES and judge reports. |
| [Packaging.py](Packaging.py) | Standard-library-only Kaggle bootstrap and verified result restoration. |
| [Plots.py](Plots.py) | Configurable Matplotlib reports over saved statistics; no smoothing. |
| [Report.py](Report.py) | ID-aligned numeric tables and reproducible, model-only evaluation reports. |
| [Results.py](Results.py) | Readable exports and notebook previews, including existing evaluation runs. |
| [Runtime.py](Runtime.py) | Sequential local-model experiments with append-only prediction segments. |

[Documentation catalogue](../../../docs/README.md)
