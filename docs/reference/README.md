# Source API catalogue

This static index describes the maintained source signatures. Read the linked implementation and docstrings for validation, failure behavior and framework-specific dynamic values.

## [code/local_backup/App.py](../../code/local_backup/App.py)

- `main(command_arguments: Any=None) -> int`: Parse command options and execute this component's command-line operation.

## [code/local_backup/Runtime.py](../../code/local_backup/Runtime.py)

- `configure_environment() -> None`: Select local model/cache locations before importing inference libraries.
- `read_access(setup: str) -> dict[str, str]`: Read the runtime settings whitelist with process values taking precedence.
- `check_space(values: dict[str, str]) -> str`: Validate the configured Space identifier and discover its remote judge endpoint.
- `preflight(values: dict[str, str], setup: str) -> dict[str, Any]`: Check runtime files, package versions and the remote endpoint without model inference.
- `class RemoteJudge`: Adapt the hosted judge endpoint to the shared scoring interface and tokenizer contract.
  - `__init__(self, values: Any, base: Any) -> None`
  - `count_input_tokens(self, messages: list[dict[str, str]]) -> int`
  - `predict_messages(self, *, model_key: str, messages: list[dict[str, str]], deadline: float) -> Any`
- `load_service(judge: Any, values: Any, setup: str) -> Any`: Load service.

## [code/local_backup/Score.py](../../code/local_backup/Score.py)

- `main(command_arguments: Any=None) -> Any`: Parse command options and execute this component's command-line operation.

## [code/offline_research/aes/RegressorEvaluator.py](../../code/offline_research/aes/RegressorEvaluator.py)

- `class RegressorEvaluator`: Manage regressor evaluator through the operations defined by this component.
  - `get_metrics_per_criterion(self, y_true: Any, y_pred: Any, criteria: Any) -> Any`
  - `plot_loss_curve(self, loss_history: Any, save_path: Any) -> None`
  - `save_predictions_table(self, y_true: Any, y_pred: Any, criteria: Any, save_path: Any, *, essay_ids: Any=None) -> None`
  - `run_full_evaluation(self, y_true: Any, y_pred: Any, criteria: Any, bands: Any, save_dir: Any, loss_history: Any=None, *, essay_ids: Any=None, report_config: Any=None) -> Any`

## [code/offline_research/aes/Training.py](../../code/offline_research/aes/Training.py)

- `aes_options(config: Any=None) -> Any`: Aes options.
- `metrics_report(truth: Any, predictions: Any, train_labels: Any) -> Any`: Baseline constants are chosen from train; test is used only for metrics.
- `run_aes(train: Any, test: Any, experiment: Any, result_root: Any, *, config: Any=None, train_model: bool=True) -> Any`: One path for PrepareData.py and the Kaggle CLI; never resplit or sample train.

## [code/offline_research/TrainAes.py](../../code/offline_research/TrainAes.py)

- `main() -> Any`: Parse command options and execute this component's command-line operation.

## [code/offline_research/data/Analysis.py](../../code/offline_research/data/Analysis.py)

- `class PreparedDataAnalyzer`: Manage prepared data analyzer through the operations defined by this component.
  - `__init__(self, frame: Any) -> None`
  - `summary(self) -> dict[str, Any]`
  - `source_summary(self) -> dict[str, Any]`
  - `profiles(self, *, with_ids: bool=False) -> Any`
  - `plot_score_distributions(self, output: Any) -> Any`

## [code/offline_research/data/DatasetTransformer.py](../../code/offline_research/data/DatasetTransformer.py)

- `class DatasetTransformer`: Manage dataset transformer through the operations defined by this component.
  - `__init__(self, datasets_paths: Any, selection: Any=None, *, save_intermediate_datasets: bool=False) -> None`
  - `prepare(self, *, limit_per_split: Any=None) -> Any`
  - `save_processed(self, prepared: Any, output: Any) -> Any`
  - `prepare_from_extracted(self, release: Any, *, allow_partial: bool=False) -> Any`
  - `prepare_and_save(self, output: Any, *, limit_per_split: Any=None) -> Any`
  - `load_processed(output: Any, source: str='combined', *, allow_partial: bool=False) -> Any`
  - `extract_score(self, text: str, criterion: str, next_criterion: str | None=None) -> float | None`
  - `load_hugging_face_datasets(self) -> Any`
  - `transform_hugging_face_dataset(self) -> Any`
  - `load_transformed_hugging_face_dataset(self) -> Any`
  - `fill_missing_scores(self, dataset: Any) -> Any`

## [code/offline_research/data/Profiles.py](../../code/offline_research/data/Profiles.py)

- `_score_profiles(frame: Any, criteria: Any, lower: Any, upper: Any, *, with_ids: Any) -> dict[str, Any]`: Score profiles.
- `count_score_profiles(frame: Any, criteria: Any, lower: int=0, upper: int=9) -> Any`: Count score profiles.
- `score_profile_ids(frame: Any, criteria: Any, lower: int=0, upper: int=9) -> Any`: Score profile ids.

## [code/offline_research/data/RawDataAnalyzer.py](../../code/offline_research/data/RawDataAnalyzer.py)

- `class RawDataAnalyzer`: Analyzes the raw dataset to give a hint where the preprocess should start first.
  - `__init__(self, dataset: pd.DataFrame, dataset_name: str) -> None`
  - `__get_dataset_shape(self) -> tuple[int, int]`
  - `print_dataset_shape(self) -> None`
  - `__check_null_values(self) -> bool`
  - `print_rows_with_null(self) -> None`
  - `print_column_names(self) -> None`
  - `plot_score_distributions(self, criteria: list[str], save_dir: str) -> None`

## [code/offline_research/data/Runtime.py](../../code/offline_research/data/Runtime.py)

- `submission_text(value: Submission) -> str`: Submission text.
- `submission_records(frame: pd.DataFrame) -> list[dict[str, Any]]`: Submission records.
- `evaluation_samples(values: SubmissionBatch) -> list[tuple[str | int, Submission, str | None]]`: Evaluation samples.
- `prediction_record(sample_id: Any, submission: Any, manifest: Any, answer: Any) -> Any`: Prediction record.
- `legacy_prediction(scores: Any) -> Any`: Legacy prediction.
- `export_reference(frame: Any, path: Any) -> Any`: Export reference.
- `convert_to_llm_dataset(frame: Any, *, test_data: bool=False) -> Any`: Convert to llm dataset.

## [code/offline_research/data/Selection.py](../../code/offline_research/data/Selection.py)

- `selection_options(config: Any) -> Any`: Selection options.
- `_select_ids(groups: Any, limit: Any, seed: Any) -> Any`: Shared memberships consume every matching quota, including adjacent targets.
- `select_dataset(frame: Any, profiles: Any, options: Any) -> Any`: Keep source splits; retain valid unselected rows with a separate status.

## [code/offline_research/data/Writer.py](../../code/offline_research/data/Writer.py)

- `validate_output(output: Any, paths: Any) -> Any`: Validate output.
- `save_prepared(prepared: Any, output: Any, paths: Any, *, selection: Any=None, save_intermediate_datasets: bool=False) -> Any`: Manifest is written last; a failed rebuild cannot expose a stale release.

## [code/offline_research/EvaluateModels.py](../../code/offline_research/EvaluateModels.py)

- `main(command_arguments: Any=None) -> int`: Parse command options and execute this component's command-line operation.

## [code/offline_research/evaluation/Config.py](../../code/offline_research/evaluation/Config.py)

- `fingerprint(value: Any) -> Any`: Fingerprint.
- `report_options(options: Any=None) -> Any`: Report options.
- `load_config(path: Any, root: Any) -> Any`: Load config.

## [code/offline_research/evaluation/Context.py](../../code/offline_research/evaluation/Context.py)

- `asset_paths(root: Any, config: Any) -> dict[str, Any]`: Asset paths.
- `nearest_references(matrix: Any, records: Any, query_vector: Any, *, essay_id: str | None, essay_hash: Any, reference_count: int) -> Any`: Nearest references.
- `prepare_contexts(paths: Any, frame: Any, experiment: Any, *, device: str='cpu', batch_size: int=8, reference_count: int=1) -> Any`: Prepare contexts.
- `class PreparedAES`: Manage prepared aes through the operations defined by this component.
  - `__init__(self, context: Any) -> None`
  - `predict_essay(self, essay: str) -> Any`
- `class PreparedRetriever`: Manage prepared retriever through the operations defined by this component.
  - `__init__(self, context: Any) -> None`
  - `retrieve(self, essay: str, *, essay_id: str | None=None, timeout: float | None=None) -> dict[str, Any]`

## [code/offline_research/evaluation/LLMAggregator.py](../../code/offline_research/evaluation/LLMAggregator.py)

- `class LLMAggregator`: Aggregates multiple JSON files containing LLM evaluation outputs,
  - `__init__(self, evaluation_folder_path: str) -> None`
  - `__load_files(self) -> None`
  - `__extract_scores(self, model_output: dict) -> dict`
  - `__compute_mode_for_file(self, file_data: list) -> dict`
  - `__clean_nulls(self) -> None`
  - `__clean_model_files(self, model: str, files: list) -> None`
  - `__clean_file(self, model: str, file_idx: int, file_data: list, mode_dict: dict) -> None`
  - `__clean_item(self, item: Any, mode_dict: dict) -> Any`
  - `__clean_answer_field(self, answer: dict, mode_dict: dict) -> None`
  - `__fill_missing_values(self, item: dict, mode_dict: dict) -> dict`
  - `__compute_stats(self) -> pd.DataFrame`
  - `__save_to_csv(self, output_path: str) -> None`
  - `__save_all_to_csv(self, output_path: str) -> None`
  - `run(self, output_path: str=PATHS['results']['llm_aggregated']) -> None`

## [code/offline_research/evaluation/LLMEvaluator.py](../../code/offline_research/evaluation/LLMEvaluator.py)

- `class LLMEvaluator`: Manage llmevaluator through the operations defined by this component.
  - `__init__(self, results_folder_path: Any, y_true_path: Any, dpi: Any=None, aes_expert_metrics: Any=None, *, aes_predictions: Any=None, report_config: Any=None) -> None`
  - `load_all(self, run_order: Any=None) -> None`
  - `_get_all_models(self) -> Any`
  - `_get_true_pred(self, run_df: Any, crit: Any) -> Any`
  - `_qwk(self, actual: Any, predicted: Any) -> Any`
  - `_mae(self, actual: Any, predicted: Any) -> Any`
  - `_rmse(self, actual: Any, predicted: Any) -> Any`
  - `_pearson_corr(self, actual: Any, predicted: Any) -> Any`
  - `_spearman_corr(self, actual: Any, predicted: Any) -> Any`
  - `plot_metrics(self, save_dir: Any) -> Any`

## [code/offline_research/evaluation/Metrics.py](../../code/offline_research/evaluation/Metrics.py)

- `integer_bands(values: Any) -> Any`: Integer bands.
- `metric_values(truth: Any, predictions: Any) -> Any`: Metric values.

## [code/offline_research/evaluation/Packaging.py](../../code/offline_research/evaluation/Packaging.py)

- `sha256_file(path: Any) -> Any`: Sha256 file.
- `inventory_id(files: Any) -> Any`: Inventory id.
- `safe_relative(name: Any) -> Any`: Safe relative.
- `validate_inventory(files: Any) -> None`: Validate inventory.
- `copy_verified(source: Any, destination: Any, digest: Any) -> None`: Copy verified.
- `extract_verified(archive: Any, destination: Any, files: Any, prefix: str='') -> None`: Extract verified.
- `prepare_project(input_root: Any, work_root: Any, manifest_path: Any=None, expected_bundle_id: Any=None) -> Any`: Prepare project.
- `restore_export(archive_path: Any, work_root: Any) -> Any`: Restore export.

## [code/offline_research/evaluation/Plots.py](../../code/offline_research/evaluation/Plots.py)

- `_axis_limits(values: Any, metric: Any) -> Any`: Leave 5% of the data span above and below the plotted mean points.
- `render_plots(tables: Any, plan: Any, output: Any, options: Any) -> dict[str, Any]`: Render plots.

## [code/offline_research/evaluation/Report.py](../../code/offline_research/evaluation/Report.py)

- `reference_frame(frame: Any) -> Any`: Reference frame.
- `aligned_results(records: Any, truth: Any, plan: Any) -> Any`: Aligned results.
- `report_tables(frame: Any, truth: Any, plan: Any) -> dict[str, Any]`: Report tables.
- `write_report(records: Any, truth: Any, plan: Any, output: Any, options: Any=None, *, model_loads: Any=()) -> Any`: Write report.

## [code/offline_research/evaluation/Results.py](../../code/offline_research/evaluation/Results.py)

- `export_results(run: Any, destination: Any=None) -> Any`: Export results.
- `show_results(run: Any, destination: str='/kaggle/working/IELTS_Evaluation_Results') -> Any`: Show results.

## [code/offline_research/evaluation/Runtime.py](../../code/offline_research/evaluation/Runtime.py)

- `unique_name() -> Any`: Unique name.
- `write_new_json(path: Any, value: Any) -> None`: Write new json.
- `environment() -> dict[str, Any]`: Environment.
- `initialize_run(root: Any, config: Any, output_root: Any, resume: Any=None) -> Any`: Initialize run.
- `prepare_run_context(run: Any, paths: Any, frame: Any, experiment: Any, manifest: Any, runtime: Any, preparer: Any=None) -> Any`: Prepare run context.
- `read_run_context(run: Any, manifest: Any) -> Any`: Read run context.
- `read_segment(path: Any, *, tolerate_tail: bool=True) -> Any`: Read segment.
- `prediction_records(run: Any) -> list[Any]`: Prediction records.
- `job_key(record: Any) -> Any`: Job key.
- `infer_models(root: Any, config: Any, run: Any, frame: Any, contexts: Any, manifest: Any, *, max_jobs: Any=None, judge_factory: Any=None) -> Any`: Infer models.
- `report_run(run: Any, contexts: Any, manifest: Any, options: Any) -> Any`: Report run.
- `export_run(run: Any) -> Any`: Export run.

## [code/offline_research/llm/LLMFineTuner.py](../../code/offline_research/llm/LLMFineTuner.py)

- `class LLMFineTuner`: Universal trainer for fine-tuning language models using LoRA or QLoRA.
  - `__init__(self, model_name: str, max_len: int=1536, mode: str='lora') -> None`
  - `apply_lora(self) -> None`
  - `_build_prompt(self, text: Submission, scores: list[float]) -> str`
  - `build_dataset(self, essays: Sequence[Submission], scores: list[list[float]]) -> Dataset`
  - `train(self, x_train: Sequence[Submission], y_train: list[list[float]]) -> None`
  - `evaluate_llm(self, x_test: Sequence[Submission], y_test: list[list[float]], output_file: str=PATHS['results']['llm_predictions']) -> list[dict]`

## [code/offline_research/llm/LLMResponseObtainer.py](../../code/offline_research/llm/LLMResponseObtainer.py)

- `class LLMResponseObtainer`: A class to obtain responses from LLMs for essay evaluation.
  - `__init__(self, SERVICE_TOKEN: str, max_new_tokens: int=256, *, setup: str='base', context_provider: Any=None) -> None`
  - `load_model_and_tokenizer(self, model_url: str) -> None`
  - `_messages(self, submission: Any) -> Any`
  - `__evaluate_essay_locally(self, essay: Submission) -> dict | None`
  - `get_response_locally(self, number_of_evaluations: int, test_dataset: SubmissionBatch) -> None`
  - `__evaluate_essay_hugging_face(self, model_url: str, essay: Submission) -> dict | None`
  - `get_responses_huggin_face(self, model_url: str, number_of_evaluations: int, test_dataset: SubmissionBatch) -> None`
  - `__connect_to_openai(self, model_url: str) -> None`
  - `__evaluate_essay_openai(self, essay: Submission) -> Any`
  - `get_responses_openai(self, model_url: str, number_of_evaluations: int, test_dataset: SubmissionBatch) -> None`

## [code/offline_research/PrepareData.py](../../code/offline_research/PrepareData.py)

- `load_data(config: Any) -> Any`: Load data.
- `main(*, prepare_only: bool=False, selection_manifest: Any=None, run_directory: Any=None) -> Any`: Parse command options and execute this component's command-line operation.

## [code/offline_research/notebooks/BuildBundle.py](../../code/offline_research/notebooks/BuildBundle.py)

- `build_bundle(prepared_release: Any=None, *, allow_partial: bool=False, output_directory: Any=None) -> Path`: Build bundle.

## [code/offline_research/notebooks/RunChecks.py](../../code/offline_research/notebooks/RunChecks.py)

- `main() -> Any`: Parse command options and execute this component's command-line operation.

## [code/offline_research/panda/Comparison.py](../../code/offline_research/panda/Comparison.py)

- `run_comparison(train: Any, experiment: Any, output: Any, profile: Any, selection: Any, generate: Any, *, limit: Any=None, retry_failed: bool=False) -> Any`: generate(records, variant) uses one shared model and identical decoding for both prompts.

## [code/offline_research/panda/Contracts.py](../../code/offline_research/panda/Contracts.py)

- `generation_profile(config: Any=None) -> dict[str, Any]`: Generation profile.
- `result_path(output: Any, essay_id: str | None) -> Any`: Result path.
- `row_input(row: Any) -> dict[str, Any]`: Row input.
- `run_manifest(train: Any, experiment: Any, profile: Any) -> dict[str, Any]`: Run manifest.
- `parse_insights(completion: Any, record: Any, profile: Any, *, allow_partial: bool=False) -> Any`: Validate insights by default; allow_partial returns unvalidated data for failed records only.
- `load_result(output: Any, row: Any, manifest: Any) -> Any`: Load result.

## [code/offline_research/panda/Corpus.py](../../code/offline_research/panda/Corpus.py)

- `build_corpus(train: Any, experiment: Any, generation_directory: Any, output: Any, profile: Any, encoder: Any, *, batch_size: int=32) -> Any`: Build corpus.

## [code/offline_research/panda/Generation.py](../../code/offline_research/panda/Generation.py)

- `pilot_order(train: Any) -> Any`: Cover source/score profiles first, then fill by stable ID hash; always retain full membership.
- `run_insights(train: Any, experiment: Any, output: Any, profile: Any, generate_batch: Any, *, limit: Any=None, retry_failed: bool=False, essay_ids: Any=None) -> Any`: Single writer. A batch returns one completion or ValueError per input, in order.
- `class HFInsightGenerator`: Loads pinned weights only on the first missing essay; generate never truncates the prompt.
  - `__init__(self, profile: Any) -> None`
  - `__call__(self, record: Any, *, prompt_variant: Any=None) -> Any`
  - `generate_batch(self, records: Any, *, prompt_variant: Any=None) -> list[Any]`

## [code/offline_research/panda/InsightGenerator.py](../../code/offline_research/panda/InsightGenerator.py)

- `class PandaInsightGenerator`: Manage panda insight generator through the operations defined by this component.
  - `__init__(self, service_token: str) -> None`
  - `build_insight_prompt(self, essay: str, score: np.ndarray, task_prompt: str | None=None, *, structured: bool=False, simple: bool=False, original: bool=False, explicit_bands: bool=False) -> str`
  - `retrieve_insights(self, essays: np.ndarray, scores: np.ndarray, num_insights: int=1, batch_size: int=12, submissions: Any=None) -> None`
  - `save_panda_insights(self, insights: Any, save_path: Any, append: bool=False, write_header: bool=True) -> None`

## [code/offline_research/panda/Panda.py](../../code/offline_research/panda/Panda.py)

- `class PANDA`: Expose the existing PANDA workflow through its original class name.

## [code/offline_research/panda/PandaModel.py](../../code/offline_research/panda/PandaModel.py)

- `class PandaModel`: Manage panda model through the operations defined by this component.
  - `__init__(self, service_token: str) -> None`
  - `load_model_and_tokenizer(self, model_url: str, *, revision: str | None=None) -> None`

## [code/offline_research/panda/RetrievalEvaluator.py](../../code/offline_research/panda/RetrievalEvaluator.py)

- `class PandaRetrievalEvaluator`: Manage panda retrieval evaluator through the operations defined by this component.
  - `load_transormer_and_tokenizer(self, transformer_url: str='BAAI/bge-base-en-v1.5', revision: str | None=None) -> None`
  - `load_panda_insights(self, path: str) -> None`
  - `build_retrieval_index(self, essays: np.ndarray | pd.DataFrame, *, experiment_manifest: Any=None) -> None`
  - `find_similar_essays(self, train_essay: Submission, num_similar: int=1) -> Any`
  - `save_results_json(self, new_entry: Any, save_path: Any) -> None`
  - `build_evaluation_messages(self, essay: Submission, similar_insights: Any) -> list[dict[str, str]]`
  - `evaluate_essay(self, essay: Submission, num_similar: int=1) -> Any`
  - `evaluate_llm(self, essays: np.ndarray, num_similar: int=1) -> None`

## [code/offline_research/ManageCorpus.py](../../code/offline_research/ManageCorpus.py)

- `data(prepared: Any) -> Any`: Data.
- `rpc_from_env(current_arguments: Any) -> Any`: Rpc from env.
- `verify_rpc(corpus: Any, rpc: Any, *, query_count: int=5) -> dict[str, Any]`: Compare true RPC and local cosine on identical saved vectors, with numerical tie tolerance.
- `main(command_arguments: Any=None) -> Any`: Parse command options and execute this component's command-line operation.

## [code/offline_research/tools/BuildAesKaggle.py](../../code/offline_research/tools/BuildAesKaggle.py)

- `build(output: Any) -> Any`: Build.

## [code/offline_research/tools/BuildEvaluationKaggle.py](../../code/offline_research/tools/BuildEvaluationKaggle.py)

- `notebook_document(bundle_id: Any=None) -> dict[str, Any]`: Notebook document.
- `build_bundle(output_root: Any=None) -> Any`: Build bundle.

## [code/offline_research/tools/BuildPandaKaggle.py](../../code/offline_research/tools/BuildPandaKaggle.py)

- `prepare_project(input_root: Any, working_root: Any, bundle_path: Any=None, expected_sha256: Any=None, *, expected_files: Any=None) -> Any`: Prepare project.
- `notebook_token(use_hf_token: bool, secret_name: Any) -> Any`: Read an explicitly enabled Kaggle secret without printing or persisting it.
- `configure_generation(project_root: Any, model_id: Any, model_revision: Any, run_name: Any, token: Any, prompt_variant: Any=None, batch_size: int=None) -> Any`: Write a separate profile with a pinned revision; keep credentials out of artifacts.
- `notebook(bundle_hash: Any, hashes: Any) -> dict[str, Any]`: Notebook.
- `build(output: Any, prepared: Any=None) -> Any`: Build.
- `main(command_arguments: Any=None) -> Any`: Parse command options and execute this component's command-line operation.

## [code/offline_research/tools/BuildStage1Space.py](../../code/offline_research/tools/BuildStage1Space.py)

- `build_bundle(output_root: Path | None=None) -> tuple[Path, Path]`: Copy only approved runtime files, preserving the Stage 0 artifacts.

## [code/offline_research/tools/PromptImage.py](../../code/offline_research/tools/PromptImage.py)

- `prompt_to_diploma_image(output_path: str=PATHS['results']['prompt_image'], width: int=1600, padding: int=80, bg_color: str='#f8f8f5ff', text_color: str='#1e1e1e', accent_color: str='#038f6aaf') -> None`: Render the IELTS prompt using Pillow's built-in scalable font.
- `draw_aes_architecture(save_path: str=PATHS['results']['aes_architecture_image'], num_heads: int=4) -> None`: Clean thesis-ready AES architecture diagram (horizontal layout).

## [code/public_demo/App.py](../../code/public_demo/App.py)

- `main() -> None`: Restore pinned runtime inputs and launch the autonomous ZeroGPU scoring interface.

## [code/public_demo/Audit.py](../../code/public_demo/Audit.py)

- `main() -> None`: Parse command options and execute this component's command-line operation.

## [code/public_demo/SourceBuilder.py](../../code/public_demo/SourceBuilder.py)

- `write_json(path: Path, value: dict[str, Any]) -> None`: Serialize the supplied value to a UTF-8 JSON document.
- `prepare_artifacts(destination: Path) -> dict[str, Any]`: Copy the accepted AES/corpus inputs and record their immutable content hashes.
- `prepare_source(destination: Path, *, artifact_repo: str, artifact_revision: str, artifact_manifest_sha256: str) -> dict[str, Any]`: Build the explicit public Space file set and pin its private runtime-artifact release.

## [code/public_demo/CpuWorker.py](../../code/public_demo/CpuWorker.py)

- `serve(connection: Any) -> None`: Process CPU-model commands until the parent closes the worker connection.
- `class AESProxy`: Forward AES predictions to the separately owned CPU model process.
  - `__init__(self, worker: Any) -> None`
  - `predict_essay(self, essay: str) -> Any`
- `class EncoderProxy`: Forward embedding requests while retaining the accepted encoder profile.
  - `__init__(self, worker: Any, profile: Any) -> None`
  - `encode(self, texts: Any) -> Any`
- `class CPUWorker`: Own one spawned CPU model process and its synchronous command connection.
  - `__init__(self) -> None`
  - `exchange(self, operation: str, payload: Any) -> Any`
  - `load_aes(self, path: Any) -> Any`
  - `load_encoder(self, profile: Any) -> Any`
  - `close(self) -> None`

## [code/public_demo/Deployment.py](../../code/public_demo/Deployment.py)

- `main() -> None`: Parse command options and execute this component's command-line operation.

## [code/public_demo/Runtime.py](../../code/public_demo/Runtime.py)

- `file_hash(path: Path) -> str`: Calculate a file's SHA-256 digest for artifact and deployment verification.
- `restore_artifacts(root: Path, deployment: dict[str, Any]) -> Path`: Restore one immutable release; never obtain artifacts from a request.
- `validate_payload(payload: dict[str, Any], model_key: str) -> float`: Keep the existing stage5_judge payload and optionally carry a deadline.
- `class HostedJudge`: Adapt a ZeroGPU generation callback to the shared scoring interface.
  - `__init__(self, judge: Any, generate: Any) -> None`
  - `count_input_tokens(self, messages: list[dict[str, str]]) -> int`
  - `predict_messages(self, *, model_key: str, messages: list[dict[str, str]], deadline: float) -> Any`

## [code/public_demo/RuntimeChecks.py](../../code/public_demo/RuntimeChecks.py)

- `class DeploymentTests`: Manage deployment tests through the operations defined by this component.
  - `payload(self) -> dict[str, Any]`
  - `test_legacy_payload_still_works(self) -> None`
  - `test_waiting_does_not_restart_the_request_deadline(self) -> None`
  - `test_external_payload_rejects_invalid_budgets_and_shapes(self) -> None`
  - `test_provider_titles_preserve_quota_and_timeout_results(self) -> None`
  - `test_public_bundle_contains_only_source_and_imports_in_isolation(self) -> None`

## [code/public_demo/Verify.py](../../code/public_demo/Verify.py)

- `call(base: Any, session: Any, name: Any, data: Any, session_hash: Any, observer: Any=None) -> Any`: Call.
- `score(base: Any, session: Any, task: Any, essay: str, language: str='en') -> dict[str, Any]`: Evaluate one Task/Essay submission and return a bounded success or failure result.

## [code/public_demo/VerifyCpu.py](../../code/public_demo/VerifyCpu.py)

- `main() -> None`: Parse command options and execute this component's command-line operation.

## [code/public_demo/VerifyFailures.py](../../code/public_demo/VerifyFailures.py)

- `class EncoderDouble`: Manage encoder double through the operations defined by this component.
  - `__init__(self, profile: Any) -> None`
  - `encode(self, texts: Any) -> Any`
- `class JudgeDouble`: Manage judge double through the operations defined by this component.
  - `__init__(self) -> None`
  - `count_input_tokens(self, messages: list[dict[str, str]]) -> int`
  - `predict_messages(self, **keyword_arguments: Any) -> dict[str, Any]`
- `main() -> None`: Parse command options and execute this component's command-line operation.

## [code/public_demo/VerifyQueue.py](../../code/public_demo/VerifyQueue.py)

- `main() -> None`: Parse command options and execute this component's command-line operation.

## [code/public_demo/VerifyRestart.py](../../code/public_demo/VerifyRestart.py)

- `main() -> None`: Parse command options and execute this component's command-line operation.

## [code/shared/ielts_judge/Access.py](../../code/shared/ielts_judge/Access.py)

- `class LauncherError`: Report launcher failures to the caller.
  - `__init__(self, code: str, message: str) -> None`
- `read_access(setup: str, *, root: Any) -> Any`: Read the runtime settings whitelist with process values taking precedence.

## [code/shared/ielts_judge/aes/Inference.py](../../code/shared/ielts_judge/aes/Inference.py)

- `write_json(path: Any, value: Any) -> None`: Serialize the supplied value to a UTF-8 JSON document.
- `save_bundle(model: Any, save_dir: Any, *, provenance: Any=None) -> Any`: Publish a new directory only after every bundle component was written.
- `read_manifest(save_dir: Any) -> Any`: Read manifest.
- `load_bundle(save_dir: Any, *, device: Any=None) -> Any`: No pretrained downloads or partial state loading are permitted here.

## [code/shared/ielts_judge/aes/OrdinalRegression.py](../../code/shared/ielts_judge/aes/OrdinalRegression.py)

- `positive_integer(value: Any, name: Any) -> Any`: Validate and return a strictly positive integer without coercion.
- `validate_texts(texts: Any) -> None`: Require a nonempty English essay string for each requested model input.
- `token_lengths(tokenizer: Any, texts: Any, batch_size: int=128) -> Any`: Count complete inputs, including special tokens, without truncation.
- `length_summary(lengths: Any, max_length: int) -> dict[str, Any]`: Summarize original token lengths and count inputs exceeding the model limit.
- `class EssayDataset`: Tokenize essay examples and attach the four ordinal training targets.
  - `__init__(self, texts: Any, targets: Any, *, tokenizer: Any, max_length: int=512) -> None`
  - `__len__(self) -> int`
  - `__getitem__(self, current_index: Any) -> Any`
- `class OrdinalHead`: Manage ordinal head through the operations defined by this component.
  - `__init__(self, hidden_dim: Any, num_classes: Any, dropout: float=0.2) -> None`
  - `forward(self, sample: Any) -> Any`
- `class OrdinalRegressionModel`: Manage ordinal regression model through the operations defined by this component.
  - `__init__(self, num_classes: int, num_tasks: int, *, encoder: PreTrainedModel) -> None`
  - `forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor`
- `class OrdinalRegression`: Manage ordinal regression through the operations defined by this component.
  - `__init__(self, num_classes: int=10, num_tasks: int=4, device: Any=None, *, encoder_name: Any=ENCODER_ID, encoder_revision: Any=ENCODER_REVISION, encoder: PreTrainedModel | None=None, tokenizer: PreTrainedTokenizerBase | None=None, max_length: int=512, learning_rate: float=2e-05, seed: int=42) -> None`
  - `_mapping(mapping: Any, num_classes: int=10) -> None`
  - `ordinal_encode(current_target: Any, band2idx: Any, num_classes: Any) -> Any`
  - `ordinal_encode_multi(y_df: Any, band2idx: Any, num_classes: Any) -> Any`
  - `ordinal_decode(logits: Any, idx2band: Any) -> Any`
  - `ordinal_decode_multi(logits: Any, idx2band: Any) -> Any`
  - `fit(self, texts: Any, y_ord: Any, epochs: int=3, batch_size: int=8) -> None`
  - `predict_logits(self, texts: Any, batch_size: int=8) -> Any`
  - `predict(self, texts: Any, idx2band: Any=None, batch_size: int=8) -> list[list[int]]`
  - `predict_essay(self, essay: str) -> dict[str, Any]`
  - `save_bundle(self, save_dir: Any, provenance: Any=None) -> Any`
  - `load_bundle(cls, save_dir: Any, device: Any=None) -> Any`
  - `save_model_and_tokenizer(self, model_wrapper: Any, save_dir: Any) -> Any`
  - `load_model_and_tokenizer(save_dir: Any, num_classes: int=10, device: Any=None) -> Any`

## [code/shared/ielts_judge/Criteria.py](../../code/shared/ielts_judge/Criteria.py)

- `load_criteria() -> dict[str, dict[str, Any]]`: Load and validate criterion names and band descriptors from YAML.
- `format_score_schema(criterion_ids: tuple[str, ...]=CRITERION_IDS) -> str`: Build the JSON score schema from the configured criterion identifiers.
- `format_rubric() -> str`: Render the shared, abbreviated rubric without loading a model.

## [code/shared/ielts_judge/data/Experiment.py](../../code/shared/ielts_judge/data/Experiment.py)

- `evaluation_options(config: Any=None) -> Any`: Evaluation options.
- `_validate_texts(frame: Any) -> None`: Validate texts.
- `_records_hash(frame: Any) -> Any`: Include content and labels, so the same IDs cannot hide changed records.
- `validate_reference_corpus(frame: Any, experiment_manifest: Any) -> None`: Enforce the full train described by a prepared experiment, before embedding.
- `prepare_experiment(train: Any, test_pool: Any, *, config: Any=None, source: str='combined', saved_manifest: Any=None) -> Any`: Return selected test and its passport; never change train or source splits.

## [code/shared/ielts_judge/data/Preparation.py](../../code/shared/ielts_judge/data/Preparation.py)

- `file_hash(path: Any) -> str`: Calculate a file's SHA-256 digest for artifact and deployment verification.
- `text_hash(text: str) -> Any`: Text hash.
- `valid_score(value: object) -> TypeGuard[int | float]`: Valid score.
- `rounded_overall(values: Any) -> Any`: Rounded overall.
- `typed_frame(rows: Any) -> Any`: Typed frame.
- `source_paths(root: Any) -> dict[str, Any]`: Source paths.
- `_nonstandard_number(value: Any) -> None`: Nonstandard number.
- `_records(path: Any, source: Any, limit: Any) -> Iterator[Any]`: Records.
- `_criterion(value: Any, name: Any, issues: Any, warnings: Any, source: Any) -> Any`: Criterion.
- `extract_sources(paths: Any, extract_score: Any, limit_per_split: Any=None) -> Any`: Read the four explicit files; preserve rows before duplicate resolution.
- `_exclude(frame: Any, index: int, reason: Any) -> None`: Exclude.
- `_signature(value: Any) -> Any`: Signature.
- `resolve_duplicates(extracted: Any) -> Any`: Select Chillies low bands, then resolve conflicts within the selected sources.
- `class PreparedDataset`: Store the prepared dataset values shared by this component's callers.
  - `split(self, name: Any) -> Any`
- `validate_splits(train: Any, test: Any) -> None`: Reject missing labels, duplicate IDs and essay leakage at the loader boundary.
- `load_prepared(output: Any, source: str='combined', *, allow_partial: bool=False) -> Any`: Load only verified accepted splits. Loading never rebuilds source files.
- `legacy_labels(frame: Any) -> Any`: One explicit adapter to legacy AES criterion order; no dataset mutation.

## [code/shared/ielts_judge/Database.py](../../code/shared/ielts_judge/Database.py)

- `retrieval_k() -> Any`: Read the shared inference setting at call time, without loading models.
- `class RetrievalError`: RPC transport, permissions, release or response failure; never an empty retrieval.
  - `__init__(self, message: str, code: str='retrieval_failure', *, diagnostics: Any=None) -> None`
- `class EncodingError`: The query encoder failed before a database request was made.
- `class SupabaseRPC`: Manage supabase rpc through the operations defined by this component.
  - `__init__(self, url: Any, key: Any, *, timeout: float | None=30) -> None`
  - `call(self, function: Any, parameters: Any, *, timeout: float | None=None) -> Any`
- `import_corpus(corpus: Any, rpc: Any, *, batch_size: int=50) -> Any`: The database owns transactional batches, immutable rows and readiness; reruns are idempotent.
- `_import_batch(rpc: Any, corpus_id: Any, batch: Any) -> None`: Import batch.
- `class ReferenceRetriever`: Does not log or persist submissions; references expose only source dataset labels.
  - `__init__(self, rpc: Any, manifest: Any, encoder: Any) -> None`
  - `retrieve(self, essay: str, *, essay_id: str | None=None, reference_count: int | None=None, threshold: Any=None, timeout: float | None=None) -> dict[str, Any]`

## [code/shared/ielts_judge/llm/Generation.py](../../code/shared/ielts_judge/llm/Generation.py)

- `class GenerationResult`: Completion and diagnostics owned by one call; no parsing or persistence.
- `generate_response(model: Any, tokenizer: Any, messages: list[dict[str, str]], *, max_new_tokens: int, generation_settings: Mapping[str, Any], criterion_ids: tuple[str, ...]=CRITERION_IDS, input_token_limit: int | None=None, deadline: float | None=None) -> GenerationResult`: Generate once using an already-loaded model, without truncating input.

## [code/shared/ielts_judge/llm/JsonEosBlocker.py](../../code/shared/ielts_judge/llm/JsonEosBlocker.py)

- `class JsonEosBlocker`: Block configured EOS tokens until the generated score JSON is complete.
  - `__init__(self, tokenizer: PreTrainedTokenizerBase, prompt_length: int, eos_tokens: list[int], criterion_ids: tuple[str, ...]=CRITERION_IDS) -> None`
  - `__call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor`
  - `_is_complete(self, response: str) -> bool`

## [code/shared/ielts_judge/llm/Judge.py](../../code/shared/ielts_judge/llm/Judge.py)

- `class Judge`: One prepared model profile; runtime owns loading, placement and concurrency.
  - `__init__(self, model_key: str, model: Any, tokenizer: Any, registry: ModelRegistry | None=None) -> None`
  - `from_pretrained(cls, model_key: str, *, device: str='cpu', token: str | None=None, registry: ModelRegistry | None=None) -> 'Judge'`
  - `predict(self, *, model_key: str, task_prompt: str, essay: str, context: JudgeContext | None=None) -> dict`
  - `count_input_tokens(self, messages: list[dict[str, str]]) -> int`
  - `predict_messages(self, *, model_key: str, messages: list[dict[str, str]], deadline: float | None=None) -> dict`

## [code/shared/ielts_judge/Localization.py](../../code/shared/ielts_judge/Localization.py)

- `translate(language: str, key: Any, **values: Any) -> Any`: Translate.

## [code/shared/ielts_judge/ModelRegistry.py](../../code/shared/ielts_judge/ModelRegistry.py)

- `class ModelProfile`: Store the model profile values shared by this component's callers.
- `class ModelRegistry`: Read and validate one immutable server-side allowlist.
  - `__init__(self, path: str | Path=REGISTRY_PATH) -> None`
  - `require_enabled(self, model_key: str) -> ModelProfile`
  - `enabled_profiles(self) -> tuple[ModelProfile, ...]`

## [code/shared/ielts_judge/panda/Artifacts.py](../../code/shared/ielts_judge/panda/Artifacts.py)

- `canonical(value: Any) -> Any`: Canonical.
- `sha256(value: Any) -> Any`: Sha256.
- `_unique_object(pairs: Any) -> Any`: Unique object.
- `parse_json(text: str) -> Any`: Parse json.
- `unwrap_completion(text: str) -> Any`: Remove exactly one complete outer Markdown fence; never extract/repair JSON fragments.
- `parse_completion(text: str) -> Any`: Parse completion.
- `is_placeholder(text: str) -> bool`: Is placeholder.
- `adjacent_band_labels(band: int) -> dict[str, str]`: Name explanation directions using the supplied integer band and its real neighbors.
- `parse_legacy_completion(text: str, labels: Any, *, explicit_bands: bool=False, allow_partial: bool=False) -> Any`: Parse text insights; allow_partial retains unvalidated diagnostic fields without fixing captions.
- `read_json(path: Any) -> Any`: Read json.
- `write_json(path: Any, value: Any) -> None`: Single-writer atomic replacement; an interrupted write leaves the prior file intact.
- `positive_int(value: Any, name: Any) -> Any`: Positive int.
- `digest(value: Any, name: str='SHA256') -> str`: Calculate the SHA-256 digest of a file without loading the whole file into memory.
- `revision(value: Any) -> Any`: Revision.
- `scores(value: Any) -> dict[str, Any]`: Scores.
- `insights(value: Any, essay: str, labels: Any, *, require_quote: bool=True) -> Any`: Insights.

## [code/shared/ielts_judge/panda/Corpus.py](../../code/shared/ielts_judge/panda/Corpus.py)

- `embedding_profile(config: Any=None) -> dict[str, Any]`: Embedding profile.
- `vector(value: Any) -> Any`: Vector.
- `class Encoder`: One encoder/profile for documents and queries, with explicit lengths and right truncation.
  - `__init__(self, profile: Any, *, device: str='cpu') -> None`
  - `encode(self, texts: Any) -> Any`
- `validate_record(record: Any, schema_version: str='panda-corpus-v1') -> Any`: Validate record.
- `class Corpus`: Validate the complete artifact on open; stream records instead of retaining all vectors in RAM.
  - `__init__(self, path: Any) -> None`
  - `records(self) -> Iterator[Any]`

## [code/shared/ielts_judge/Paths.py](../../code/shared/ielts_judge/Paths.py)

- `_resolve_paths(value: Any, root: Any=PROJECT_ROOT) -> Any`: Resolve paths.
- `load_paths(project_root: Any=None) -> Any`: Resolve a selected checkout's configuration, including notebook bundles.

## [code/shared/ielts_judge/Prompts.py](../../code/shared/ielts_judge/Prompts.py)

- `class ReferenceContext`: Keep the source scores and raw insights of one retrieved training essay.
- `class JudgeContext`: Select the rubric, references and essay-only AES guidance for one prompt.
- `_public_scores(scores: Mapping[str, float]) -> dict[str, float]`: Map legacy context names once without changing the supplied values.
- `submission_parts(submission: Any) -> tuple[str, str]`: Only Task/Essay enter inference; dataset target columns are never serialized.
- `build_judge_messages(task_prompt: str, essay: str, context: JudgeContext | None=None) -> list[dict[str, str]]`: Reference IDs/tasks/essays stay in metadata; preserve all supplied text.

## [code/shared/ielts_judge/Scores.py](../../code/shared/ielts_judge/Scores.py)

- `public_scores(value: Any, *, legacy: bool=False) -> dict[str, Any]`: Normalize integer-valued scores without rounding or filling fields.
- `parse_scores(text: str) -> Any`: Allow whitespace and one whole JSON fence; reject ambiguous/partial output.

## [code/shared/ielts_judge/Scoring.py](../../code/shared/ielts_judge/Scoring.py)

- `class ScoringError`: Carry a public failure code and message from a scoring stage.
  - `__init__(self, code: str, message: str) -> None`
- `class JudgeQuotaError`: The hosting adapter recognized a provider quota rejection.
- `class ScoringLimits`: Define validated character and elapsed-time limits for one scoring request.
  - `__post_init__(self) -> None`
- `class ScoringService`: Runtime supplies loaded components; request data is never persisted or cached.
  - `__init__(self, judge: Any, *, aes: Any=None, retriever: Any=None, setup: str='rubric_panda_aes', limits: ScoringLimits | None=None, versions: dict[str, Any] | None=None) -> None`
  - `_prompt(self, request: dict[str, Any]) -> Any`
  - `_generate(self, request: dict[str, Any]) -> Any`
  - `_parse(self, request: dict[str, Any]) -> Any`
  - `score(self, *, model_key: str, task_prompt: str, essay: str, essay_id: str | None=None) -> dict[str, Any]`

## [code/shared/ielts_judge/ScoringRuntime.py](../../code/shared/ielts_judge/ScoringRuntime.py)

- `load_scoring_service(judge: Any, *, aes_release: str | Path, corpus_release: str | Path, project_root: str | Path, supabase_url: str, supabase_key: str, setup: str='rubric_panda_aes', limits: Any=None, cpu_worker: Any=None) -> ScoringService`: Load once on CPU, preserving the trained AES and pinned embedding profile.

## [code/shared/ielts_judge/Ui.py](../../code/shared/ielts_judge/Ui.py)

- `load_examples() -> Any`: Load examples.
- `render_result(result: Any, language: str=DEFAULT_LANGUAGE) -> Any`: Translate presentation only; preserve the original service result.
- `render_presentation(language: str, result: Any=None, status_key: str='ready') -> Any`: Prepare messages once so switching language needs no server request.
- `build_demo(service: Any) -> Any`: Connect the shared scoring service to the localized Gradio submission interface.

## [Manage.py](../../Manage.py)

- `environment(env_file: Path | None=None) -> dict[str, str]`: Load supported settings without replacing values supplied by the process.
- `source_archive(destination: Path) -> int`: Create a source distribution without credentials, data, weights or local policy.
- `class CommandRunner`: Run component commands with one explicit interpreter, checkout and environment.
  - `__init__(self, python: Path, values: dict[str, str]) -> None`
  - `run(self, arguments: list[str]) -> int`
  - `install(self, target: str, dry_run: bool) -> int`
  - `test(self, suite: str) -> int`
- `create_parser() -> argparse.ArgumentParser`: Define the project command interface independently of command execution.
- `main(command_arguments: list[str] | None=None) -> int`: Dispatch an explicit project command through its selected execution context.

## [RuntimeAssets.py](../../RuntimeAssets.py)

- `digest(path: Any) -> str`: Calculate the SHA-256 digest of a file without loading the whole file into memory.
- `restore(source: Any) -> Any`: Validate compatible runtime artifacts before copying missing files into this checkout.
- `main() -> int`: Parse command options and execute this component's command-line operation.
