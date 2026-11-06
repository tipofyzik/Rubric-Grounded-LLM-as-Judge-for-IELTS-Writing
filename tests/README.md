# Tests

Verify meaningful scoring, dataset, notebook and deployment contracts. Manage.py test selects supported suites. Portable checks require no private artifacts; full suites need configured local fixtures. Test doubles are explicit and do not establish model quality.

| File | Responsibility |
|---|---|
| [RunPortable.py](RunPortable.py) | Run offline CPU contract checks from a checkout without private artifacts. |
| [TestAES.py](TestAES.py) | Real tiny Transformers training, bundles and the full-train consumer contract. |
| [TestAESKaggle.py](TestAESKaggle.py) | Executable notebook bootstrap and source syntax, without pip or GPU jobs. |
| [TestAppRuntime.py](TestAppRuntime.py) | Permanent project runtime configuration; no credentials or model calls. |
| [TestDatabase.mjs](TestDatabase.mjs) | Configuration, dependency or supporting source input. |
| [TestDatasetConsumers.py](TestDatasetConsumers.py) | Task and ID propagation through inference, SFT, retrieval and evaluation. |
| [TestDatasetIntegration.py](TestDatasetIntegration.py) | Data contracts exercised through real CSV, JSONL and Parquet I/O. |
| [TestDatasetSelection.py](TestDatasetSelection.py) | Profile IDs and final split quotas, including overlapping target groups. |
| [TestDistribution.py](TestDistribution.py) | User-facing installation and component boundaries of the source distribution. |
| [TestExperiment.py](TestExperiment.py) | Experiment selection, persisted replay, consumer routing and corpus exclusions. |
| [TestJudge.py](TestJudge.py) | Judge contracts: real prompts/registry/generation, no pretrained downloads. |
| [TestModelEvaluation.py](TestModelEvaluation.py) | Portable evaluation checks: known scores, real files, report contents and resume. |
| [TestPandaBatching.py](TestPandaBatching.py) | Batch boundaries, recovery, left padding and real local Transformers generation. |
| [TestPandaCorpus.py](TestPandaCorpus.py) | PANDA membership, recovery, artifacts, HTTP boundaries, packaging and real encoder checks. |
| [TestRetrievalRecovery.py](TestRetrievalRecovery.py) | Read-only HTTP recovery and safe service diagnostics; actual loopback HTTP. |
| [TestScoringCLI.py](TestScoringCLI.py) | Launcher and HTTP-stream handling checks; no real model/GPU calls. |
| [TestScoringService.py](TestScoringService.py) | Scoring contracts: real render/parser/LangChain; model doubles are explicit. |
| [TestStage1Space.py](TestStage1Space.py) | Check the manual probe and its exact deployable core without hosted access. |
| [TestUILocalization.py](TestUILocalization.py) | UI language behavior with real Gradio and an explicit scoring-service double. |

[Documentation catalogue](../docs/README.md)
