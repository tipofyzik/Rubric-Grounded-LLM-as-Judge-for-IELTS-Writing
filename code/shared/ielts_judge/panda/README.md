# Panda

Read compatible corpus records and manifests, validate raw insights, and load the pinned embedding profile. Runtime retrieval uses the accepted corpus; building a new corpus is a research action.

| File | Responsibility |
|---|---|
| [__init__.py](__init__.py) | PANDA insight generation and retrieval evaluation. |
| [Artifacts.py](Artifacts.py) | Strict, versioned PANDA artifacts; no model loading or network at import. |
| [Corpus.py](Corpus.py) | Build and verify portable, full-train PANDA corpora before SQL ingestion. |

[Documentation catalogue](../../../../docs/README.md)
