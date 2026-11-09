# Evaluation

Compare Qwen prompts and the essay scorer on the same test essays.

## Run

Use a prepared dataset and matching AES and PANDA releases. The default run uses CUDA. [Settings](../notebooks/ModelEvaluationConfig.yaml)

From the repository root, run `python Manage.py research evaluate --mode all` for predictions and a report. For a pilot, set `evaluation.sample_size` to 10 or 20 in a copy of the settings and pass `--config PATH`.

To split the run, use the directory printed by `prepare` in the next two commands.

```sh
python Manage.py research evaluate --mode prepare
python Manage.py research evaluate --mode infer --resume-run "PATH_TO_RUN" --max-jobs 100 --export
python Manage.py research evaluate --mode report --resume-run "PATH_TO_RUN" --export
```

Rerun `infer` with the same settings to continue. `report` uses saved predictions and writes tables and charts under `reports/`. Add `--export` to save a results ZIP.

## Resume on Kaggle

Set `RESUME_RUN` to the run directory to continue within a session. To continue in another session, attach the results ZIP alongside the input Dataset and set `RESUME_ARCHIVE`.

[`EvaluateModels.py`](../EvaluateModels.py) · [Reporting](Report.py)
