# Verification scope

## Portable CPU checks and GitHub Actions

```powershell
& '.\.venv\Scripts\python.exe' -B code/tests/run_portable.py
```

The same command is used by `.github/workflows/model-evaluation.yml` on Ubuntu
and Python 3.11, with CPU torch 2.10.0 and the pinned evaluation dependencies.
Datasets and PEFT are installed for the shared fine-tuned-inference contract test.
No pretrained model, private dataset, weights, credentials or GPU is required.
The workflow uploads `.cache/checks/checks.log` and `results.json` as evidence.

The selected modules cover integer metrics, undefined values, aligned coverage,
plot/report/export contents, verified input restoration, append-only resume,
shared prompt/parser/EOS contracts, judge routing/budgets, ScoringService errors
and HTTP retrieval recovery. Tiny models are initialized locally to exercise
real CPU generation; model doubles are labelled in the individual tests.

The legacy response checks now use Task/Essay records, public score keys, strict
parsing, greedy inference and the existing 256-token default. They continue to
check prompt-token exclusion, EOS handling, raw diagnostics and interrupted
persistence. Production inference is unchanged by this test update.

A passing local run does not establish remote CI success. Cite a GitHub Actions
run URL and its exact commit when reporting remote CI; a branch name alone is
insufficient. This portable selection is not the entire local suite.

The [verified GitHub Actions run](https://github.com/tipofyzik/Rubric-Grounded-LLM-as-Judge-for-IELTS-Writing/actions/runs/42132015393)
executed 89 tests with zero failures, errors or skips for commit
`dde73bb360a3f9ef23446a532643f01dd745e6d0`. Its downloaded `portable-checks`
artifact contains the test log and machine-readable summary. The prepared
workspace's separate main suite passed all 213 tests on 6 November 2026.

## Prepared-workspace regression checks

```powershell
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s code/tests -p 'Test*.py' -v
& '.\.venv\Scripts\python.exe' -B kaggle/run_checks.py
```

These checks additionally use local source/release fixtures, historical notebook
builders and UI dependencies. Retain their logs and record any skip explicitly.
A clean public checkout does not provide those ignored inputs. Do not present
the portable workflow as executing this full prepared-workspace coverage.

The historical notebook's plotting regression uses integer prepared inputs and
the shared reporter's new output directory. Historical fractional reference
tables are explicitly rejected without rounding and retained unchanged. This
keeps historical fixtures separate from the current integer-score benchmark.
The prepared-workspace Kaggle suite completed 40 tests: 39 passed and one
directory-symlink check was skipped because this Windows host does not permit
creating that symlink. This suite is not run by the portable workflow.

## Real-model evidence

The completed Kaggle baseline comprises 1635 essays and 9810 Qwen jobs; see the
[report](model_evaluation_report.md) and [protocol](model_evaluation_protocol.md).
Portable CPU checks are not evidence of Qwen grading quality, Kaggle CUDA
compatibility, model robustness or hosted-service availability. Additional real
repeatability/robustness runs are pending. Public-service evaluation is deferred.
