# Standalone model evaluation

This workflow evaluates AES and local Hugging Face judge models in a **new Kaggle
notebook with a new private Dataset**. Public service evaluation is deferred.
No HF Space, Supabase RPC, training or reference-insight generation is called.

## Files and input package

- `kaggle/ModelEvaluation.ipynb`: new generic notebook.
- `kaggle/ModelEvaluationConfig.yaml`: experiment, resource and plot options.
- `kaggle/ModelEvaluationModels.yaml`: separate pinned research model registry.
- `code/evaluate_models.py`: local-model CLI used by the notebook.
- `code/tools/build_evaluation_kaggle.py`: verifies the accepted inputs and creates
  a unique bundle directory and a matching notebook pinned to its bundle ID.

Build from the prepared project environment:

```powershell
& '.\.venv\Scripts\python.exe' -B code/tools/build_evaluation_kaggle.py
```

The printed directory under `results/stage8/kaggle/` contains the new
`model_evaluation_bundle.zip`, `ModelEvaluationBundleManifest.json`, a pinned
`ModelEvaluation.ipynb` and upload instructions. Each build creates a new directory.
Source datasets, accepted AES/corpus releases and old Kaggle inputs are preserved.
The package contains the verified ielts-v3 prepared release, accepted AES weights,
corpus v2 and embeddings, shared inference code and configuration. Model downloads
use pinned Qwen/BGE revisions. Secrets and historical prediction tables are excluded.

## Kaggle setup

1. Create a **new private Dataset**, suggested name `ielts-writing-model-evaluation`.
2. Upload the ZIP and its matching `ModelEvaluationBundleManifest.json` together.
   The bootstrap also accepts an unpacked ZIP beside that manifest.
3. Create a **new notebook**, suggested title `IELTS Writing Model Evaluation`,
   and import the matching pinned notebook from the same build directory.
4. Attach the new Dataset and enable a GPU and Internet.
5. The bootstrap requires an unambiguous evaluation manifest and validates the
   notebook/bundle ID, archive inventory and every file hash. It creates a new
   working project rather than reusing old notebook directories.

The notebook creates a new venv inheriting the host CUDA PyTorch. Evaluation
dependencies are installed into that environment with an exact constraint for
the host torch version. The model command runs in a fresh Python subprocess.
The actual Python/package/CUDA/device versions are saved with the experiment.
The accepted Qwen GPU pilot and full baseline have completed. A changed model
or execution environment needs its own compatibility check.

## Experiment selection

The initial registry contains the same pinned Qwen2.5 1.5B profile as the demo.
Adding a research profile here does not change the public app registry.
The six setups are `base`, `rubric`, `panda`, `rubric_panda`, `panda_aes` and
`rubric_panda_aes`; AES is always reported separately as an essay-only transformer.

Default selection is the whole 1,635-row test pool, seed 42, one repeat. A full
six-setup pass is 9,810 judge jobs, excluding AES/context preparation. For the
first technical pilot, edit the notebook's `CONFIG_YAML` to set sample_size to
10 or 20. A changed sample, model, setup or inference setting creates a separate
experiment; it does not replace the full configuration or any prior result.

Test labels are only used for reports. AES and BGE context are calculated once
and reused across setups and repeats. Retrieval uses the accepted full-train
corpus locally, excludes query ID/hash, preserves raw insights/source scores,
and records query truncation and selected references. User inputs are not
silently truncated; the shared scoring core enforces full prompt budgets.

The existing test pool has prior evaluation history and is not certified as an
untouched final holdout. Tuning needs a separately declared development/final
protocol. Site/source labels are not certified examiner marks.

## Configurable plots and tables

Edit `report.plots` in the YAML. An empty list disables figures, while all numeric
tables and coverage reports are still saved. The default report contains exactly
seven figures for one model: adjacent accuracy (within one band), MAE, RMSE,
Pearson, Spearman, QWK and the original source-to-prediction calibration grid.
PNG at 200 DPI is the default; set `report.formats: [png, svg]` to add vector SVG.
WEBP is also supported. Other plot types remain available only by explicit selection.
`report.layout: grid` selects the default 2x2 criterion grids; `single` writes one
figure per criterion for metric comparisons and calibration. Single filenames
end in `_TR`, `_CC`, `_LR` or `_GRA`. One model produces seven grid figures or
28 single figures. The four corresponding single figures use the grid's limits.

Metric figures use the original 2x2 criterion grid, model-colored solid trends,
black-edged markers, across-repeat STD when measured, a red dashed AES baseline,
prompt separators and separate global model/prompt legends. The model legend
contains the four criterion-specific AES values, calculated on common-valid IDs.
Calibration uses setup-colored curves, a black dashed identity line, shared axes
with integer ticks 0 through 9 and a global setup legend; it has no AES curve.

Y-axis limits use the minimum and maximum plotted mean points, including AES,
across all four criteria before choosing grid or single layout. Both ends get
the same visual margin: padding = 0.05 * (max - min), lower = min - padding,
upper = max + padding. Axis padding can extend beyond a metric's mathematical
range, so values at 0, 1 or 9 remain clear of the frame; actual values and their
validation are unchanged. Zero/identity reference lines and STD error bars do
not determine these point-based limits. Calibration uses its setup curve points,
not the identity-line endpoints. Constant points use a nonzero margin based on
their magnitude, falling back to the metric's range at zero. Empty plots use the
metric's range.
Empty calibration bands remain missing, and a single repeat does not produce
measured STD.

Available names:

| Name | Content |
|---|---|
| metric_comparison | Original four-criterion comparison with global model/prompt legends and numeric AES baseline |
| metric_bars | Separate model comparisons with setup bars and numeric AES baseline |
| metric_heatmap | Model/setup by criterion metric values, including AES |
| calibration | Original calibration curves: mean prediction by source score; filenames MODEL_calibration_grid |
| calibration_by_prediction | Mean source score by predicted band, including band 9 |
| bias | Mean prediction minus source score by source band |
| error_histogram | Mean fractions at fixed integer errors -9 through +9 |
| error_histogram_std | Across-repeat STD of those fractions; requires repeated runs |
| error_ridge | Unsmoothed offset error distributions |
| confusion_matrix | Integer 10x10 count and row-normalized matrices |
| score_distribution | Source and predicted score distributions on common valid IDs |
| coverage | Valid, failed and unprocessed fractions of all planned requests |
| prompt_comparison | Metric changes relative to base on common valid IDs (optional) |
| repeatability | Criterion-score agreement for essays with at least two valid repeats |
| latency | Judge wall-time distributions in this model execution environment |

There is **no smoothing** and **no +/-0.5 accuracy**. Exact and within-one-band
accuracy are distinct. MAE, MSE, RMSE, QWK, Pearson, Spearman and mean bias are
supported. Undefined metrics carry null plus a reason; failures are not imputed.
Sample STD is between repeated metrics, not a confidence interval. For one
defined run STD/repeatability are unmeasured and error bars are omitted.

`report.formats` selects PNG/SVG/WEBP; `report.dpi` controls raster resolution.
`report.style.rc` accepts Matplotlib rcParams; `aes`, `models` and `setups`
accept line/errorbar style dictionaries. For example:

```yaml
report:
  plots: [metric_comparison, calibration]
  metrics: [within_one_accuracy, mae, rmse, pearson, spearman, qwk]
  formats: [png] # Add svg for vector output.
  dpi: 200
  layout: grid # Use single for one image per criterion with the same shared limits.
  style:
    aes: {color: red, linestyle: '--', linewidth: 2}
    models: {qwen25_1p5b: {color: navy}}
    setups: {base: {color: gray}}
    rc: {font.size: 11, axes.grid: true}
```

Numeric tables include predictions, per-run metrics, summaries, formatted
display values, coverage, failure counts, calibration/bias groups, fixed-bin
errors, confusion counts, reference distributions, prompt comparisons, repeatability
and common valid IDs. Quality comparisons/AES lines use IDs valid in **every**
planned model/setup/repeat, so the baseline is evaluated on the same population.
Own-valid metrics are also exported. Each denominator is explicit. Incomplete
runs remain labelled incomplete; absent common observations do not become zeros.

## Resume and export

The command prints the unique run directory. Set notebook `RESUME_RUN` to that
directory to continue within a session. `MAX_JOBS` limits new jobs without
changing the experiment's planned denominator. Completed failures are retained
and are not automatically retried. Only missing jobs are resumed.

Predictions use new append-only JSONL segments and flush/fsync after each record.
An interrupted, uncommitted final line is preserved and ignored; completed
malformed records or duplicate job keys stop processing. A killed request whose
result was not committed may run again. Use one writer session per run.

Each invocation with `--export` creates a new ZIP. Keep/download the printed ZIP
with a readable model/sample/setup/repeat/date filename. The last notebook cell
also creates a separate archive under `/kaggle/working/IELTS_Evaluation_Results`,
displays a download link and previews the original calibration curves.
Repeated exports add a numeric suffix instead of replacing previous archives.
and the notebook with outputs. To continue in another Kaggle session, attach the
result ZIP as input and set `RESUME_ARCHIVE`; restoration verifies every exported
file into a new working directory. Keep the matching model input Dataset attached.

The CLI can be used directly as well:

```powershell
& '.\.venv\Scripts\python.exe' -B code/evaluate_models.py --mode prepare --config PATH_TO_CONFIG
& '.\.venv\Scripts\python.exe' -B code/evaluate_models.py --mode infer --config PATH_TO_CONFIG --resume-run PATH_TO_RUN --max-jobs 100 --export
& '.\.venv\Scripts\python.exe' -B code/evaluate_models.py --mode report --config PATH_TO_CONFIG --resume-run PATH_TO_RUN --export
```

`prepare` runs real AES/BGE inference, without a judge. `infer` loads selected
models locally; it never calls the hosted service. `report` reads exported
context, reference and predictions without loading models or requiring model
weights/source datasets. Changed plot settings create a new report directory.
Resume for inference validates inputs, selected IDs, settings, registry, core
source hashes and package/Python versions. GPU/backend changes also require a
separate run rather than mixing inference environments. Model/tokenizer startup
times are saved separately in `model_loads.csv`; they may include downloads.

## Verification and scope

```powershell
& '.\.venv\Scripts\python.exe' -B code/tests/TestModelEvaluation.py -v
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s code/tests -p 'Test*.py' -v
```

Portable tests cover known integer metrics, undefined values, failures/coverage,
ID/release mismatch, plot types and numeric AES legends, fixed bins/band 9,
append-only resume, unique exports, and ZIP/unpacked input restoration. Model
doubles in persistence tests are explicitly not a real Qwen quality benchmark.

The Kaggle GPU pilot and full 1,635-essay comparison (9,810 judge jobs, one repeat)
have been completed and analysed. Current results, raw-run locations and remaining
protocol/repeatability/robustness/reporting work are recorded in the
[protocol](docs/model_evaluation_protocol.md), [baseline report](docs/model_evaluation_report.md)
and [verification guide](docs/verification.md). The prepared workspace additionally
retains `context/model_evaluation_context.md` and
`results/stage8/analysis/2026-11-04-full1635/analysis.md`.
Plot-only changes can use report mode with the saved full run; another inference
run is not required. Public service latency/queue/restart/quota checks remain
**deferred by the user's decision** and are not run by this workflow.
