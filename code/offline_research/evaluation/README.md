# Evaluation

Do a rubric, similar examples and a first estimate help small language models score essays more accurately? This code compares prompt variants on the same test essays and measures how closely each set of scores matches the dataset.

## Run

Use a verified Parquet release and matching AES and PANDA release descriptors. [`ModelEvaluationConfig.yaml`](../notebooks/ModelEvaluationConfig.yaml) lists the input paths, model profiles, six prompt setups and report settings. The default judge device is CUDA. Model weights may need to be downloaded.

From the repository root, use `python Manage.py research evaluate --mode all` to prepare context, run inference and generate a report in one command. For a small first run, set `evaluation.sample_size` to 10 or 20 in a copy of the YAML and pass `--config PATH_TO_CONFIG` on each invocation.

To run each command separately, prepare the context once and use the printed run directory for `PATH_TO_RUN`.

```sh
python Manage.py research evaluate --mode prepare
python Manage.py research evaluate --mode infer --resume-run "PATH_TO_RUN" --max-jobs 100 --export
python Manage.py research evaluate --mode report --resume-run "PATH_TO_RUN" --export
```

`prepare` saves the run manifest and AES/BGE context. Rerun `infer` with the same settings to continue missing jobs. `report` reads the saved run without model weights and writes a new report under `reports/`. Set `report.plots` in the YAML to choose charts. An empty list keeps the numeric tables and skips charts. `--export` prints the path of a new results ZIP.

## Resume on Kaggle

Use the notebook generated alongside the input ZIP and its manifest. Set `RESUME_RUN` to the printed run path to continue within a session. In another session, attach the exported results ZIP alongside the matching model input Dataset and set `RESUME_ARCHIVE`. The notebook verifies the archive and restores the run into a new working directory.

[`EvaluateModels.py`](../EvaluateModels.py) runs the comparison and saves predictions. [`Report.py`](Report.py) and [`Plots.py`](Plots.py) turn predictions into tables and charts.

The research index in the source repository describes the release descriptors and links to the full evaluation report.
