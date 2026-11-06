"""Exercise the shipped notebook cells locally without downloading model weights."""

import ast
from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile


project_root = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
temporary_folder = project_root / "code/offline_research/notebooks/.cache"
temporary_folder.mkdir(parents=True, exist_ok=True)
for variable in ("TEMP", "TMP", "TMPDIR"):
    os.environ[variable] = str(temporary_folder)
tempfile.tempdir = str(temporary_folder)
sys.dont_write_bytecode = True
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(temporary_folder / "TestCache/matplotlib")
os.environ["HF_HOME"] = str(temporary_folder / "TestCache/huggingface")
os.environ["TORCH_HOME"] = str(temporary_folder / "TestCache/torch")
os.environ["XDG_CACHE_HOME"] = str(temporary_folder / "TestCache")

import numpy as np
import pandas as pd
import yaml


class TestKaggleNotebook(unittest.TestCase):
    """Run actual notebook cells with local paths and explicitly mocked generation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Extract the runtime ZIP into the reusable Kaggle test workspace.

        Returns:
            None: This operation returns no value.
        """
        cls.original_directory = Path.cwd()
        cls.output_folder = (project_root / "code/offline_research/notebooks/artifacts/tests/notebook").resolve()
        if not cls.output_folder.is_relative_to((project_root / "code/offline_research/notebooks/artifacts/tests").resolve()):
            raise ValueError("Test workspace escapes Kaggle artifacts")
        if cls.output_folder.exists():
            shutil.rmtree(cls.output_folder)
        cls.output_folder.mkdir(parents=True)
        cls.local_project = cls.output_folder / "project"
        cls.bundle_path = project_root / "code/offline_research/notebooks/artifacts/tests/bundle/ielts_kaggle_bundle.zip"
        notebook = json.loads((project_root / "code/offline_research/notebooks/KaggleNotebook.ipynb").read_text(encoding="utf-8"))
        cls.code_cells = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
        cls.base_namespace = {}
        cls.bootstrap(cls.base_namespace, cls.bundle_path, cls.local_project)
        cls.base_configuration = yaml.safe_load((cls.local_project / "RunConfig.yaml").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore the caller's working directory while retaining test artifacts.

        Returns:
            None: This operation returns no value.
        """
        os.chdir(cls.original_directory)

    @classmethod
    def bootstrap(cls, namespace: dict, source: Path, destination: Path) -> None:
        """Execute the real first cell with only Kaggle mount paths replaced.

        Args:
            namespace (dict): Namespace used by this operation.
            source (Path): Source used by this operation.
            destination (Path): Destination used by this operation.

        Returns:
            None: This operation returns no value.
        """
        tree = ast.parse(cls.code_cells[0])
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "input_source":
                    node.value = ast.Constant(str(source))
                elif node.targets[0].id == "project_root":
                    node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                          args=[ast.Constant(str(destination))], keywords=[])
        exec(compile(ast.fix_missing_locations(tree), "NotebookBootstrap", "exec"), namespace)

    def setUp(self) -> None:
        """Prepare a fresh notebook namespace and a persistent per-test log.

        Returns:
            None: This operation returns no value.
        """
        self.namespace = dict(self.base_namespace)
        self.configuration = copy.deepcopy(self.base_configuration)
        # Existing CSV regression cases keep their original source-label contract.
        self.configuration["data"]["dataset_key"] = "transformed_hugging_face"
        self.configuration["run_name"] = self._testMethodName
        self.configuration["install_dependencies"] = False
        self.configuration["strict_checks"] = False
        self.configuration["data"]["sample_limit"] = 3
        self.configuration["data"]["train_limit"] = 5
        self.configuration["evaluation"]["dpi"] = 35
        self.log_path = self.output_folder / f"{self._testMethodName}.log"
        self.log_file = self.enterContext(self.log_path.open("w", encoding="utf-8"))
        self.enterContext(redirect_stdout(self.log_file))
        self.enterContext(redirect_stderr(self.log_file))

    def execute(self, number: int) -> None:
        """Execute one unmodified notebook cell in the test namespace.

        Args:
            number (int): Number used by this operation.

        Returns:
            None: This operation returns no value.
        """
        exec(compile(self.code_cells[number], f"NotebookCell{number + 1}", "exec"), self.namespace)

    def prepare(self, mode: str = "inference", dependencies: bool = True) -> None:
        """Assemble the current operation's validated inputs and metadata.

        Args:
            mode (str): Mode used by this operation.
            dependencies (bool): Dependencies used by this operation.

        Returns:
            None: This operation returns no value.
        """
        self.configuration["mode"] = mode
        (self.local_project / "RunConfig.yaml").write_text(
            yaml.safe_dump(self.configuration, sort_keys=False), encoding="utf-8"
        )
        self.execute(1)
        if not dependencies:
            return
        with patch("subprocess.run", side_effect=AssertionError("Unexpected package installation")):
            self.execute(2)
        if mode != "evaluate_existing":
            import torch
            with patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(torch.cuda, "device_count", return_value=1), \
                    patch.object(torch.cuda, "get_device_name", return_value="Mock GPU; not hardware tested"):
                self.execute(3)
        else:
            self.execute(3)

    def generate(self) -> None:
        """Exercise the real response writer with model loading and answers mocked.

        Returns:
            None: This operation returns no value.
        """
        from offline_research.llm.LLMResponseObtainer import LLMResponseObtainer
        criteria = self.namespace["criteria"]
        scores = self.namespace["test_frame"][criteria].to_numpy(dtype=float)
        answers = [dict(zip(criteria, np.rint(values * 2) / 2)) for values in scores]
        repetitions = self.configuration["inference"]["runs"]
        with patch.object(LLMResponseObtainer, "load_model_and_tokenizer") as loader, \
                patch.object(LLMResponseObtainer, "_LLMResponseObtainer__evaluate_essay_locally",
                             side_effect=answers * repetitions) as evaluator:
            self.execute(5)
        loader.assert_called_once_with(model_url=self.configuration["model"]["id"])
        self.assertEqual(self.namespace["response_obtainer"].max_new_tokens,
                         self.configuration["inference"]["max_new_tokens"])
        self.assertEqual(evaluator.call_count, len(answers) * repetitions)

    def test_notebook_and_bundle(self) -> None:
        """Check Python syntax, ZIP CRCs and the runtime-only archive contract.

        Returns:
            None: This operation returns no value.
        """
        self.assertEqual(len(self.code_cells), 9)
        for source in self.code_cells:
            ast.parse(source)
        with zipfile.ZipFile(self.bundle_path) as archive:
            self.assertIsNone(archive.testzip())
            names = set(archive.namelist())
        self.assertIn("RunConfig.yaml", names)
        self.assertIn("requirements.txt", names)
        self.assertIn("code/shared/paths.yaml", names)
        self.assertIn("code/shared/ielts_judge/__init__.py", names)
        self.assertIn("code/offline_research/data/Selection.py", names)
        self.assertFalse(any(name.startswith("code/offline_research/tools/") for name in names))
        self.assertFalse(any(name.startswith(("code/offline_research/notebooks/", "TEMP/", ".git/")) for name in names))
        self.assertFalse(any(name.endswith((".ipynb", ".md", ".pyc", ".env")) for name in names))
        manifest = json.loads(self.bundle_path.with_name("KaggleBundleManifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["archive_sha256"], hashlib.sha256(self.bundle_path.read_bytes()).hexdigest())
        with zipfile.ZipFile(self.bundle_path) as archive:
            self.assertEqual(names, set(manifest["files"]))
            for name, expected_hash in manifest["files"].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected_hash, name)
                if name.startswith("code/shared/ielts_judge/") and name.endswith(".py"):
                    self.assertEqual(archive.read(name), (project_root / name).read_bytes(), name)

    def test_builder_checks_release_and_ignores_unlisted_files(self) -> None:
        """Verify that builder checks release and ignores unlisted files.

        Returns:
            None: This operation returns no value.
        """
        sys.path.insert(0, str(project_root / "code/offline_research/notebooks"))
        from offline_research.notebooks.BuildBundle import build_bundle
        from ielts_judge.Paths import load_paths
        release = Path(load_paths(self.local_project)["datasets"]["prepared"])
        with self.assertRaisesRegex(ValueError, "Partial"):
            build_bundle(release)
        with self.assertRaisesRegex(FileNotFoundError, "manifest"):
            build_bundle(self.output_folder / "missing-release")
        extra = release / "unlisted.parquet"
        extra.write_bytes(b"stale intermediate file")
        try:
            build_bundle(release, allow_partial=True)
            with zipfile.ZipFile(self.bundle_path) as archive:
                self.assertFalse(any(name.endswith("unlisted.parquet") for name in archive.namelist()))
        finally:
            extra.unlink()

        train_file = release / "train.parquet"
        original = train_file.read_bytes()
        try:
            train_file.write_bytes(b"corrupt parquet")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_bundle(release, allow_partial=True)
        finally:
            train_file.write_bytes(original)

    def test_invalid_run_name_is_rejected(self) -> None:
        """Verify that invalid run name is rejected.

        Returns:
            None: This operation returns no value.
        """
        self.configuration["run_name"] = "../outside"
        with self.assertRaisesRegex(ValueError, "run_name"):
            self.prepare(dependencies=False)

    def test_bootstrap_zip_twice(self) -> None:
        """Unpack the ZIP twice without deleting an existing experiment artifact.

        Returns:
            None: This operation returns no value.
        """
        marker = self.local_project / "code/offline_research/notebooks/artifacts/keep.txt"
        marker.write_text("existing result", encoding="utf-8")
        self.bootstrap(self.namespace, self.bundle_path, self.local_project)
        self.bootstrap(self.namespace, self.bundle_path, self.local_project)
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing result")

    def test_bootstrap_extracted_input(self) -> None:
        """Copy an extracted runtime Dataset twice without a code/offline_research/notebooks folder.

        Returns:
            None: This operation returns no value.
        """
        source = self.output_folder / "extracted-input"
        with zipfile.ZipFile(self.bundle_path) as archive:
            archive.extractall(source)
        self.assertFalse((source / "code/offline_research/notebooks").exists())
        self.bootstrap(self.namespace, source, self.local_project)
        self.bootstrap(self.namespace, source, self.local_project)
        self.assertTrue((self.local_project / "RunConfig.yaml").is_file())

    def test_environment_and_data_repeat(self) -> None:
        """Reproduce repeated environment/data cells and preserve source labels 7.25.

        Returns:
            None: This operation returns no value.
        """
        self.prepare()
        import torch
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "device_count", return_value=0):
            self.execute(3)
        self.execute(4)
        expected = pd.read_csv(self.namespace["reference_path"])
        self.execute(4)
        pd.testing.assert_frame_equal(expected, pd.read_csv(self.namespace["reference_path"]))
        for name in ("raw_folder", "evaluation_folder", "plot_folder"):
            self.assertTrue(self.namespace[name].is_dir())
        dataset = self.namespace["dataset"]
        source_row = dataset.loc[dataset["source_row"] == 1632].iloc[0]
        self.assertEqual(source_row["lexical_resource"], 7.25)
        self.assertEqual(source_row["grammatical_range_and_accuracy"], 7.25)
        self.assertEqual(len(dataset), 10323)

    def test_same_run_replaces_previous_without_suffix(self) -> None:
        """Replace one named experiment and keep other named experiments intact.

        Returns:
            None: This operation returns no value.
        """
        self.prepare()
        original = self.namespace["run_folder"]
        marker = original / "keep.txt"
        marker.write_text("old results", encoding="utf-8")
        other = original.parent / "separate_experiment"
        other.mkdir(exist_ok=True)
        (other / "keep.txt").write_text("different run", encoding="utf-8")
        archive = original.parent.parent / f"{original.name}.zip"
        archive.write_bytes(b"old export")
        self.execute(1)
        self.assertEqual(original, self.namespace["run_folder"])
        # Parameter re-execution alone must not discard existing results.
        self.assertTrue(marker.is_file())
        self.assertFalse(original.with_name(original.name + "_2").exists())
        self.namespace["mode"] = "evaluate_existing"
        self.execute(3)
        self.assertFalse(marker.exists())
        self.assertFalse(archive.exists())
        self.assertEqual((other / "keep.txt").read_text(encoding="utf-8"), "different run")

    def test_run_symlink_cannot_remove_external_directory(self) -> None:
        """Verify that run symlink cannot remove external directory.

        Returns:
            None: This operation returns no value.
        """
        self.prepare()
        original = self.namespace["run_folder"]
        shutil.rmtree(original)
        outside = self.output_folder / "outside-run"
        outside.mkdir()
        marker = outside / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        try:
            original.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Directory symlinks are unavailable on this host")
        self.namespace["mode"] = "evaluate_existing"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.execute(3)
        self.assertTrue(marker.exists())
        original.unlink()

    def test_bootstrap_keeps_all_caches_inside_kaggle(self) -> None:
        """Verify that bootstrap keeps all caches inside kaggle.

        Returns:
            None: This operation returns no value.
        """
        cache_root = self.local_project / "code/offline_research/notebooks/.cache"
        for variable in ("TEMP", "TMP", "TMPDIR", "HF_HOME", "HF_HUB_CACHE",
                         "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "MPLCONFIGDIR",
                         "XDG_CACHE_HOME", "PIP_CACHE_DIR", "WANDB_DIR"):
            self.assertTrue(Path(os.environ[variable]).is_relative_to(cache_root), variable)
        self.assertFalse((self.local_project / "TEMP").exists())

    def test_prepared_parquet_preserves_splits_tasks_and_prediction_ids(self) -> None:
        """Verify that prepared parquet preserves splits tasks and prediction ids.

        Returns:
            None: This operation returns no value.
        """
        self.configuration['data'].update(dataset_key='prepared', allow_partial=True)
        self.prepare()
        self.execute(4)
        train, test = self.namespace['train_frame'], self.namespace['test_frame']
        self.assertTrue(train['split'].eq('train').all())
        self.assertTrue(test['split'].eq('test').all())
        self.assertIn('task_prompt', test)
        self.assertFalse(set(train.essay_hash) & set(test.essay_hash))
        self.generate()
        self.execute(6)
        predictions = self.namespace['result_frame']
        self.assertEqual(set(predictions.sample_id), set(test.essay_id))
        self.assertTrue(predictions.dataset_manifest.eq(test.attrs['manifest_sha256']).all())
        reference = pd.read_csv(self.namespace['reference_path'])
        self.assertEqual(reference['index'].tolist(), test.essay_id.tolist())

    def test_partial_prepared_parquet_is_rejected_by_default(self) -> None:
        """Verify that partial prepared parquet is rejected by default.

        Returns:
            None: This operation returns no value.
        """
        self.configuration['data'].update(dataset_key='prepared', allow_partial=False)
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'Partial'):
            self.execute(4)

    def test_custom_preprocessed_data(self) -> None:
        """Load the second bundled dataset through the custom CSV option.

        Returns:
            None: This operation returns no value.
        """
        self.configuration["data"]["dataset_key"] = "custom"
        self.prepare()
        self.execute(4)
        self.assertEqual(self.namespace["summary"]["source_rows"], 10323)
        self.assertEqual(len(self.namespace["test_frame"]), 3)

    def test_existing_dependencies_skip_pip(self) -> None:
        """Use available packages without invoking pip or a restart guard.

        Returns:
            None: This operation returns no value.
        """
        self.configuration["install_dependencies"] = True
        self.prepare(dependencies=False)
        with patch("subprocess.run") as installer:
            self.execute(2)
            self.execute(2)
        installer.assert_not_called()
        self.assertEqual(self.namespace["missing_packages"], [])

    def test_only_missing_dependency_installed(self) -> None:
        """Mock one absent package and inspect constrained installation arguments.

        Returns:
            None: This operation returns no value.
        """
        self.configuration["install_dependencies"] = True
        self.prepare(dependencies=False)
        distributions = [item for item in metadata.distributions()
                         if item.metadata["Name"].lower().replace("_", "-") != "accelerate"]
        with patch.object(metadata, "distributions", return_value=distributions), \
                patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="Mock install")) as installer:
            self.execute(2)
            self.execute(2)
        self.assertEqual(installer.call_count, 2)
        command = installer.call_args.args[0]
        self.assertEqual(command[-1], "accelerate==1.13.0")
        self.assertIn("--constraint", command)
        self.assertNotIn("restart_required", self.namespace)

    def test_inference_pipeline(self) -> None:
        """Run actual CSV/plot/export cells with only LLM generation mocked.

        Returns:
            None: This operation returns no value.
        """
        self.configuration["data"].update(dataset_key="prepared", allow_partial=True)
        self.configuration["inference"]["runs"] = 2
        self.prepare()
        self.execute(4)
        self.generate()
        self.execute(6)
        self.assertTrue(self.namespace["reports_ready"])
        self.assertEqual(len(self.namespace["result_frame"]), 6)
        self.execute(7)
        plot_folder = self.namespace["plot_folder"]
        self.assertEqual(len(list(plot_folder.rglob("*.png"))), 7)
        self.assertEqual(len(list(plot_folder.rglob("*.csv"))), 15)
        self.execute(8)
        self.execute(8)
        with zipfile.ZipFile(self.namespace["result_archive"]) as archive:
            self.assertIsNone(archive.testzip())
            self.assertTrue(any("plots/" in name and name.endswith(".png") for name in archive.namelist()))

    def test_full_rerun_has_no_stale_responses_or_duplicate_exports(self) -> None:
        """Verify that full rerun has no stale responses or duplicate exports.

        Returns:
            None: This operation returns no value.
        """
        self.configuration["evaluation"]["make_plots"] = False
        for repetitions in (2, 1):
            self.configuration["inference"]["runs"] = repetitions
            self.prepare()
            self.execute(4)
            self.generate()
            self.execute(6)
            self.execute(8)
            self.assertEqual(len(list(self.namespace["raw_folder"].glob("*.json"))), repetitions)
            self.assertEqual(len(self.namespace["result_frame"]), 3 * repetitions)
        run_folder = self.namespace["run_folder"]
        self.assertEqual(list(run_folder.parent.glob(run_folder.name + "*")), [run_folder])
        self.assertEqual(len(list(run_folder.parent.parent.glob(run_folder.name + "*.zip"))), 1)
        with zipfile.ZipFile(self.namespace["result_archive"]) as archive:
            self.assertFalse(any(name.endswith(f"{self.namespace['model_key']}_2.json") for name in archive.namelist()))

    def test_invalid_answer_stays_missing(self) -> None:
        """Keep a missing model answer explicit and skip invalid metric reports.

        Returns:
            None: This operation returns no value.
        """
        self.prepare()
        self.execute(4)
        criteria = self.namespace["criteria"]
        answers = [None, dict.fromkeys(criteria, 6.5), dict.fromkeys(criteria, 7)]
        response_path = self.namespace["raw_folder"] / f"{self.namespace['model_key']}_1.json"
        response_path.write_text(json.dumps(answers), encoding="utf-8")
        self.execute(6)
        self.execute(7)
        self.assertFalse(self.namespace["reports_ready"])
        self.assertTrue(self.namespace["result_frame"].loc[0, criteria].isna().all())
        self.assertEqual(len(self.namespace["invalid_frame"]), 1)
        self.assertFalse(list(self.namespace["plot_folder"].rglob("*.png")))
        self.execute(8)

    def test_fine_tune_orchestration(self) -> None:
        """Check training arguments and adapter/export wiring with the trainer mocked.

        Returns:
            None: This operation returns no value.
        """
        self.configuration["training"]["mode"] = "lora"
        self.prepare("fine_tune")
        self.execute(4)
        criteria = self.namespace["criteria"]
        predictions = [{"prediction_json": row.to_dict()}
                       for _, row in self.namespace["test_frame"][criteria].iterrows()]
        save_adapter = Mock(side_effect=lambda folder: Path(folder).mkdir(parents=True, exist_ok=True))
        trainer = SimpleNamespace(
            model=SimpleNamespace(save_pretrained=save_adapter),
            tokenizer=SimpleNamespace(save_pretrained=Mock()),
            train=Mock(), evaluate_llm=Mock(return_value=predictions),
            _build_prompt=Mock(side_effect=AssertionError("Optional length checks should be disabled")),
        )
        constructor = Mock(return_value=trainer)
        module = ModuleType("offline_research.llm.LLMFineTuner")
        module.LLMFineTuner = constructor
        previous_module = sys.modules.get(module.__name__)
        sys.modules[module.__name__] = module
        try:
            self.execute(5)
        finally:
            if previous_module is None:
                sys.modules.pop(module.__name__, None)
            else:
                sys.modules[module.__name__] = previous_module
        constructor.assert_called_once_with(
            model_name=self.configuration["model"]["id"], mode="lora",
            max_len=self.configuration["training"]["max_length"],
        )
        trainer.train.assert_called_once()
        self.assertEqual(len(trainer.train.call_args.args[0]), 5)
        self.assertTrue((self.namespace["run_folder"] / "adapter").is_dir())
        self.execute(6)
        self.assertEqual(self.namespace["experiment_key"], "fine_tuned_raw_result")
        self.assertEqual(len(self.namespace["result_frame"]), 3)
        self.execute(8)

    def test_relaxed_and_strict_outputs(self) -> None:
        """Accept fractional answers during a trial and retain optional grid checks.

        Returns:
            None: This operation returns no value.
        """
        self.prepare()
        self.execute(4)
        answers = [dict.fromkeys(self.namespace["criteria"], 7.25)] * 3
        response_path = self.namespace["raw_folder"] / f"{self.namespace['model_key']}_1.json"
        response_path.write_text(json.dumps(answers), encoding="utf-8")
        self.execute(6)
        self.assertTrue(self.namespace["reports_ready"])
        self.namespace["strict_checks"] = True
        self.execute(6)
        self.assertFalse(self.namespace["reports_ready"])

    def test_old_extra_raw_run_ignored(self) -> None:
        """Aggregate configured repetitions without including a stale extra file.

        Returns:
            None: This operation returns no value.
        """
        self.prepare()
        self.execute(4)
        answers = [dict.fromkeys(self.namespace["criteria"], 6)] * 3
        for number in (1, 2):
            response_path = self.namespace["raw_folder"] / f"{self.namespace['model_key']}_{number}.json"
            response_path.write_text(json.dumps(answers), encoding="utf-8")
        self.execute(6)
        self.assertEqual(len(self.namespace["result_frame"]), 3)

    def test_reference_ids_match_samples(self) -> None:
        """Persist consistent reference/prediction identifiers for the chosen rows.

        Returns:
            None: This operation returns no value.
        """
        self.prepare()
        self.execute(4)
        reference = pd.read_csv(self.namespace["reference_path"])
        samples = pd.read_csv(self.namespace["run_folder"] / "samples.csv")
        self.assertEqual(reference["index"].tolist(), samples["sample_id"].tolist())
        pd.testing.assert_frame_equal(reference[self.namespace["criteria"]], samples[self.namespace["criteria"]])
        self.assertEqual(samples["source_row"].tolist(), self.namespace["test_frame"]["source_row"].tolist())

    def test_historical_fractional_reports_are_rejected_without_rounding(self) -> None:
        """Preserve historical fractions and reject them in the integer reporter.

        Returns:
            None: This operation returns no value.
        """
        self.prepare("evaluate_existing")
        for number in (4, 5, 6):
            self.execute(number)
        reference_before = self.namespace["reference_path"].read_bytes()
        with self.assertRaisesRegex(ValueError, "integer bands.*no rounding"):
            self.execute(7)
        self.assertEqual(self.namespace["reference_path"].read_bytes(), reference_before)
        self.assertFalse(list(self.namespace["plot_folder"].rglob("*.png")))


if __name__ == "__main__":
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestKaggleNotebook)
    )
    results_folder = getattr(TestKaggleNotebook, "output_folder", temporary_folder)
    (results_folder / "Results.txt").write_text(output.getvalue(), encoding="utf-8")
    summary = {"tests": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped),
               "python": sys.version, "results": str(results_folder),
               "limitations": "Local Python 3.11; GPU detection, LLM generation and fine-tuning trainer mocked. No real model, CUDA training or Kaggle runtime tested."}
    (results_folder / "Results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output.getvalue())
    print(json.dumps(summary, indent=2))
    sys.exit(not result.wasSuccessful())
