# Panda

Own optional generation of explanations for known source scores, corpus construction/comparison and research retrieval evaluation. This differs from assigning scores to a new essay.

| File | Responsibility |
|---|---|
| [__init__.py](__init__.py) | Offline research panda workflows. |
| [Comparison.py](Comparison.py) | A resumable paired prompt experiment; this selection is never a reference corpus. |
| [Contracts.py](Contracts.py) | Full-train, resumable PANDA generation with one atomic result per essay ID. |
| [Corpus.py](Corpus.py) | Offline corpus workflow; shared runtime owns read contracts. |
| [Generation.py](Generation.py) | Offline generation workflow; shared runtime owns read contracts. |
| [InsightGenerator.py](InsightGenerator.py) | Generate PANDA criterion insights from expert-scored essays. |
| [Panda.py](Panda.py) | Compatibility facade combining PANDA generation and retrieval workflows. |
| [PandaModel.py](PandaModel.py) | Shared local Hugging Face model loading for PANDA components. |
| [RetrievalEvaluator.py](RetrievalEvaluator.py) | Retrieve PANDA insights and evaluate essays with a local judge. |

[Documentation catalogue](../../../docs/README.md)
