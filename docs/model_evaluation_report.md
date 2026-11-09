# Model evaluation: Qwen2.5-1.5B and AES

Baseline report, 6 November 2026. [Evaluation protocol](model_evaluation_protocol.md)

## Result

On the 1635-essay evaluation pool, no Qwen setup improves AES on MAE or QWK for
any criterion. The full hybrid mostly reproduces AES and is slightly worse in
aggregate. This result concerns the fixed model, prompt, corpus and k=1 setting;
it does not establish that rubric or retrieval context cannot help other models.

All 9810 Qwen jobs and 1635 AES outputs are valid. Common-valid N is 1635.
Each configuration ran once.

| Configuration | Macro QWK | Macro RMSE | Macro MAE | Exact accuracy | Within-one accuracy |
|---|---:|---:|---:|---:|---:|
| AES | 0.607192 | 1.390925 | 0.979511 | 37.37% | 74.95% |
| Base | 0.026655 | 2.127120 | 1.724312 | 15.57% | 49.54% |
| Rubric | 0.000000 | 3.161052 | 2.675688 | 14.51% | 24.65% |
| PANDA | 0.092408 | 2.234527 | 1.738991 | 21.10% | 50.69% |
| Rubric + PANDA | 0.090790 | 2.239002 | 1.742508 | 21.09% | 50.61% |
| PANDA + AES | 0.595384 | 1.426560 | 1.016361 | 35.81% | 73.67% |
| Rubric + PANDA + AES | 0.606128 | 1.393129 | 0.981040 | 37.34% | 74.91% |

Macro averages combine four criterion metrics, not an overall IELTS Writing band.
The 6540 criterion scores per configuration share the same 1635 essay IDs.

## Observed output behaviour

Rubric-only returns four nines for all 1635 essays. These values are present in
raw completions; they are not parser fallback values. Its constant predictions
make correlations undefined, while QWK against varying labels is defined as zero.

PANDA copies the complete retrieved reference score vector for 1627/1635 essays;
rubric + PANDA does so for 1632/1635. The full hybrid copies 6536/6540 AES scores
and the entire AES vector for 1632/1635 essays. Its four changed scores on three
essays all increase absolute error. PANDA + AES changes 161 essay vectors:
38 improve, 107 worsen and 16 retain the same summed absolute error.

These comparisons measure output agreement with the scores supplied in the prompts.

## Time and resources

Qwen startup takes 12.94 seconds. The 9810 jobs sum to 21061.68 seconds of job
wall time (about 5 hours 51 minutes) and 20960.77 seconds of GPU work. Mean job
wall time ranges from 2.0175 seconds (base) to 2.3514 seconds (full hybrid).
One-time context preparation takes 164.23 seconds. Peak per-job CUDA allocation
is approximately 3664 MiB. These timings describe model inference on Kaggle.

Judge prompts reach at most 2220 tokens, below the 3840 input budget; outputs
reach at most 47 tokens, below 256. All completions finish with EOS. Context
encoders truncate 25 AES inputs and 16 BGE queries at their 512-token limits.

## Reproduce the report

Use a saved evaluation run to rebuild the report from the repository root.

```sh
python Manage.py research evaluate --mode report --resume-run "PATH_TO_RUN"
```

Report mode reads the saved run and writes new tables and plots.

## Limitations

The test pool is previously used, heavily Writing9-dominated and has shared
train/test questions. Source labels are not certified examiner annotations.
See the protocol for the complete data, AES and corpus limitations.
