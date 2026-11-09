# Model evaluation protocol

Status: baseline protocol documented retrospectively on 6 November 2026.
Repeatability and robustness experiments are pending the user's choice of scope
and subsequent Kaggle runs. This document does not preregister completed work.

## Research question and scope

For Qwen2.5-1.5B-Instruct, does rubric context, retrieved PANDA context or AES
guidance improve criterion-score agreement with source labels over AES alone?
Compare AES with `base`, `rubric`, `panda`, `rubric_panda`, `panda_aes` and
`rubric_panda_aes` on identical essay IDs. This is a model experiment, separate
from public-service latency, queue, restart and quota evaluation, which is deferred.
An unfavourable hybrid result is a valid outcome; prompt tuning is a future study.

## Accepted baseline and provenance

The supplied Kaggle run is `20261104T160302Z-470168bdfc`, fingerprint
`a07f3cd74fc8fe69478b756e34fb055c90e457519e48670f252e46c9ec5e9615`.
Its matching input bundle is
`073d75318edfd3f5ea7908d61d890c4fa18cc37a61d1224471714f41a69886be`.
The later `78b8…` bundle changes report defaults and is not the provenance of
the completed baseline. Use the saved run config, not today's defaults.

| Component | Baseline specification |
|---|---|
| Data | `ielts-v3`, manifest SHA-256 `11ae1900c7742aed1a1bfa88583015c0638e13ced802130292d6577d7ee4e2e6` |
| AES | Accepted full-train ordinal RoBERTa bundle; manifest SHA-256 `d87cec3348ffb13c5fa708c26d97b88c3d13a72937e87b3b7b2cbea2a7cf6eec` |
| Judge/tokenizer | `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Prompt/parser | `task2-scores-v2` / `scores-integer-v2`; source hashes in the run manifest |
| Corpus | Raw-completion v2, ID `f71532c72bc4f3a2371598971cc2d8a9f016f69cad4a977423754e40e4541266` |
| Retrieval | Essay-only cosine, k=1, `BAAI/bge-base-en-v1.5` revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, 768 normalized float32 dimensions |
| Generation | Greedy (`do_sample=false`), 4096 context tokens, up to 256 output tokens; full rendered input limit 3840 |
| Service limits reused locally | 40,000 input characters, 120-second request deadline, one judge attempt |
| Experiment | All 1635 test essays, all six setups, one repeat, selection seed 42 |

AES was trained on all 8236 selected training essays. Its separate accepted
training release records the training provenance; the generic experiment manifest's
`aes_training_provenance: unknown` field does not replace that release evidence.
AES sees Essay only. Mapping its first output to `task_response` does not make
that estimate sensitive to Task. AES and BGE use a 512-token input limit and
record truncation; the judge's Task/Essay text is not silently truncated.

The local baseline is `results/evaluation/runs/20261104T160302Z-470168bdfc/`.
The archive inventory records 201 file hashes. The run contains selected IDs,
labels, contexts, raw predictions, environment and timings. These are private
research artifacts.

## Data and leakage boundaries

The release retains original source texts and labels after the documented
source validation, conflict removal, deduplication and profile selection.
Train/test contain 8236/1635 essays; test contains 1613 Writing9 and 22 Chillies
essays. Source labels, particularly Writing9 site/crowd scores, are not certified
examiner marks. The source imbalance limits interpretation of aggregate metrics.

Train/test IDs and normalized essay hashes are disjoint. Questions are allowed
to overlap: 322 shared question hashes and 382 test essays with a train question.
This is not a question-disjoint evaluation. All train essays belong to AES/PANDA;
test essays do not belong to the reference corpus. Query ID/hash exclusions apply.

Test labels enter metric calculations only. The judge receives Task/Essay,
optional rubric, the retrieved training example's scores/raw insights, and optional
AES predictions. Retrieved IDs, tasks and full essays stay outside the prompt.
The corpus includes accepted, failed and partial generation completions without
repair or acceptance filtering. Technical generation acceptance is not evidence
of semantic correctness of an insight.

## Metrics, denominators and failures

The output is exactly four public criterion keys with finite integer values 0-9.
Mathematically integral numbers such as `6.0` and one whole JSON Markdown fence
are accepted. Fractional, missing, duplicate or extra fields and arbitrary
surrounding text are rejected. No rounding, imputation, output repair or automatic
retry turns a failed output into a valid score.

For each criterion report MAE, MSE, RMSE, quadratic weighted kappa (fixed bands
0-9), exact accuracy, within-one-band accuracy, Pearson, Spearman and signed mean
bias. Macro summaries are arithmetic means of four criterion metrics; they are
not an official overall IELTS band. Half-band accuracy and smoothing are absent.

Own-valid metrics use valid observations for that model/setup/repeat. Common-valid
metrics and AES comparison lines use the intersection of IDs valid in every
planned model/setup/repeat, including AES. Retain planned, valid, failed and
not-run counts; never silently compare different populations. Undefined metrics
retain null with a reason, including correlations for constant predictions.
QWK can be defined as zero when predictions are constant but source labels vary.

Across-repeat summaries use metric means and sample STD (`ddof=1`); one defined
repeat leaves STD unmeasured. Repeat STD is not a confidence interval. For future
repeated runs also compare exact criterion and full-vector agreement and score
ranges. Deterministic copying of AES can be reproducible without demonstrating
meaningful robustness.

## Timing scope and environment

The baseline used Python 3.13.16, Linux, torch 2.11.0+cu130 and two visible Tesla
T4 devices. Visibility does not establish multi-GPU inference. Both judge and
context devices were configured as CUDA. Exact packages/backend flags are saved
in the run and inference environment manifests.

Model/tokenizer startup is recorded separately and may include uncached downloads.
Per-job wall time and GPU work exclude the one-time batch context preparation.
Context batch time is not per-submission service latency. GPU work and allocation
are not hosted quota billing. A later CPU-context run must not be silently resumed
into this CUDA-context inference history.

## Development and final evaluation policy

The full test pool has already been examined in pilots, the complete baseline and
historical comparisons. It is an adjustable, previously used evaluation pool,
not an untouched holdout. The present study reports the fixed baseline without
tuning or selecting a new winning configuration on this pool.

Before future tuning, specify development data and an independently reserved
final evaluation population, selection policy, fixed settings and reporting
rules before inspecting final outcomes. Reserving IDs from an already inspected
pool cannot undo its history. UI examples are not final evaluation data.

## Pending addenda owned jointly with the user

The user chooses the additional experiment sizes and runs Kaggle. The previously
discussed 100 essays x 3 repeats and 60 robustness cases remain proposals; no
sample IDs, number of repeats or custom-case manifest is fixed by this document.

Before execution, save the repeatability IDs/settings and a robustness manifest
with stable case/group IDs, original-to-variant links, exact input hashes and
expected behaviour. Cover original/paraphrased/mismatched Task, instructions
inside Essay, and short/long boundary inputs. Modified inputs do not inherit
ground-truth labels automatically; use paired behavioural analysis unless an
additional expert annotation is supplied. Correct input rejection is a recorded
outcome. Freeze these addenda before running either experiment.
