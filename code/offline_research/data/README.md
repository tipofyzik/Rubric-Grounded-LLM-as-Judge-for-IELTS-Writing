# Data

Own source ingestion, profile selection, analysis and prepared-release writing. Original source content and labels are retained according to the documented validation/deduplication rules.

| File | Responsibility |
|---|---|
| [__init__.py](__init__.py) | Offline research data workflows. |
| [Analysis.py](Analysis.py) | Audit and plots for the common Parquet schema. |
| [DatasetTransformer.py](DatasetTransformer.py) | One offline entry point for Chillies extraction and Writing9 preparation. |
| [Profiles.py](Profiles.py) | Globally disjoint criterion profiles shared by preparation and legacy analysis. |
| [RawDataAnalyzer.py](RawDataAnalyzer.py) | Python package marker. |
| [Runtime.py](Runtime.py) | Explicit adapters from verified dataset rows to model/evaluation inputs. |
| [Selection.py](Selection.py) | Deterministic profile quotas applied after source validation and deduplication. |
| [Writer.py](Writer.py) | Offline writer workflow; shared runtime owns read contracts. |

[Documentation catalogue](../../../docs/README.md)
