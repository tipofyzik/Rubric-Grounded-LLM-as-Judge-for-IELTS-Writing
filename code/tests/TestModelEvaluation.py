"""Portable evaluation checks: known scores, real files, report contents and resume."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import pandas as pd

from ielts_judge.criteria import PUBLIC_CRITERION_IDS
from ielts_judge.data.preparation import text_hash
from ielts_judge.evaluation.config import PLOT_TYPES, load_config, report_options
from ielts_judge.evaluation.context import nearest_references
from ielts_judge.evaluation.metrics import metric_values
from ielts_judge.evaluation.report import aligned_results, reference_frame, report_tables, write_report
from ielts_judge.evaluation.runtime import export_run, infer_models, prediction_records, read_segment, write_new_json
from ielts_judge.evaluation.packaging import inventory_id, prepare_project, restore_export, sha256_file
from ielts_judge.prompts import PROMPT_VERSION, CONTEXT_SETUPS
from ielts_judge.scores import OUTPUT_SCHEMA_VERSION

CRITERIA = list(PUBLIC_CRITERION_IDS)


def fixture(repeats=2):
    truth = pd.DataFrame([{"essay_id": f"test:{i}", **dict.fromkeys(CRITERIA, band)} for i, band in enumerate((0, 3, 6, 9))])
    plan = {"models": ["qwen25_1p5b"], "model_labels": {"qwen25_1p5b": "Qwen test"},
            "setups": ["base", "rubric_panda_aes"], "repeats": repeats,
            "essay_ids": truth.essay_id.tolist(), "dataset_manifest_sha256": "a" * 64}
    rows = []
    for row in truth.to_dict("records"):
        rows.append({"model_key": "aes", "setup": "aes", "repeat": 0, "essay_id": row["essay_id"],
                     "dataset_manifest": "a" * 64, "status": "ok", "scores": {c: row[c] for c in CRITERIA}})
        for setup in plan["setups"]:
            for repeat in range(repeats):
                band = min(9, row[CRITERIA[0]] + (repeat if setup == "base" else 0))
                rows.append({"model_key": "qwen25_1p5b", "setup": setup, "repeat": repeat,
                             "essay_id": row["essay_id"], "dataset_manifest": "a" * 64, "status": "ok",
                             "scores": dict.fromkeys(CRITERIA, band), "timings": {"judge_seconds": 0.1 + repeat * 0.02}})
    return truth, plan, rows


class ModelEvaluationTests(unittest.TestCase):
    def test_integer_metrics_known_values_and_undefined_reasons(self):
        values, reasons = metric_values([0, 1, 2], [1, 1, 1])
        self.assertAlmostEqual(values["mae"], 2 / 3)
        self.assertAlmostEqual(values["exact_accuracy"], 1 / 3)
        self.assertEqual(values["within_one_accuracy"], 1)
        self.assertAlmostEqual(values["qwk"], 0)
        self.assertIsNone(values["pearson"])
        self.assertIn("pearson", reasons)
        constant, reasons = metric_values([9, 9], [9, 9])
        self.assertIsNone(constant["qwk"])
        self.assertEqual(reasons["qwk"], "single identical category")
        for invalid in ([6.5], [True], [False, 1], [float("nan")], [10], ["6"]):
            with self.assertRaises(ValueError):
                metric_values(invalid, invalid)

    def test_config_selects_plot_types_and_rejects_removed_options(self):
        config = load_config(ROOT / "kaggle/ModelEvaluationConfig.yaml", ROOT)
        self.assertEqual(config["setups"], list(CONTEXT_SETUPS))
        for options in (config["report"], report_options()):
            self.assertEqual(options["metrics"], ["within_one_accuracy", "mae", "rmse", "pearson", "spearman", "qwk"])
            self.assertEqual(options["plots"], ["metric_comparison", "calibration"])
            self.assertEqual(options["formats"], ["png"])
            self.assertEqual(options["dpi"], 200)
            self.assertEqual(options["layout"], "grid")
        self.assertEqual(report_options({"layout": "single"})["layout"], "single")
        self.assertEqual(report_options({"formats": ["png", "svg"]})["formats"], ["png", "svg"])
        self.assertEqual(report_options({"plots": []})["plots"], [])
        for invalid in ({"smooth": "gaussian"}, {"metrics": ["adjacent_accuracy"]}, {"plots": ["unknown"]}, {"layout": "unknown"}):
            with self.assertRaises(ValueError):
                report_options(invalid)

    def test_coverage_and_common_ids_do_not_impute_invalid_or_missing(self):
        truth, plan, rows = fixture(1)
        rows[-1]["scores"][CRITERIA[0]] = 6.5
        rows.pop(-2)
        frame = aligned_results(rows, truth.iloc[::-1], plan)
        tables = report_tables(frame, truth, plan)
        self.assertEqual(set(tables["common_ids"].essay_id), {"test:0", "test:1", "test:2"})
        self.assertEqual(int(frame.status.eq("not_run").sum()), 1)
        self.assertEqual(int(frame.status.eq("error").sum()), 1)
        self.assertTrue(frame.loc[frame.status.eq("error"), CRITERIA].isna().all().all())
        self.assertEqual(int(tables["coverage"].planned.sum()), 12)
        self.assertEqual(int(tables["coverage"].valid.sum()), 10)

    def test_duplicate_wrong_dataset_and_reference_ids_are_rejected(self):
        truth, plan, rows = fixture(1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aligned_results(rows + [rows[0]], truth, plan)
        rows[0]["dataset_manifest"] = "wrong"
        with self.assertRaisesRegex(ValueError, "different dataset"):
            aligned_results(rows, truth, plan)
        with self.assertRaises(ValueError):
            reference_frame(pd.concat([truth, truth.iloc[:1]]))

    def test_fixed_error_bins_and_score_nine_are_counted(self):
        truth, plan, rows = fixture(1)
        tables = report_tables(aligned_results(rows, truth, plan), truth, plan)
        for _, group in tables["error_distribution"].groupby(["model_key", "setup", "repeat", "criterion"]):
            self.assertEqual(group.error.tolist(), list(range(-9, 10)))
            self.assertEqual(int(group["count"].sum()), 4)
        ninth = tables["calibration"].loc[tables["calibration"].band.eq(9)]
        self.assertTrue(ninth["count"].eq(1).all())
        empty = tables["score_response"].loc[tables["score_response"].band.eq(1)]
        self.assertTrue(empty.mean_prediction.isna().all())
        self.assertTrue(tables["metrics_summary"]["std"].isna().all())
        for _, group in tables["confusion"].groupby(["model_key", "setup", "repeat", "criterion"]):
            self.assertEqual(int(group["count"].sum()), 4)

    def test_report_tables_without_plots_and_no_overwrite(self):
        truth, plan, rows = fixture(1)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report"
            write_report(rows, truth, plan, output, {"plots": []})
            manifest = json.loads((output / "report_manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["plots"]["files"], [])
            self.assertTrue((output / "metrics_summary.csv").is_file())
            self.assertTrue((output / "prompt_comparison.csv").is_file())
            with self.assertRaises(FileExistsError):
                write_report(rows, truth, plan, output, {"plots": []})

    def test_every_plot_type_and_numeric_aes_legend(self):
        import matplotlib.pyplot as plt
        truth, plan, rows = fixture(2)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report"
            write_report(rows, truth, plan, output,
                         {"plots": list(PLOT_TYPES), "metrics": ["mae"], "formats": ["svg"], "dpi": 40})
            manifest = json.loads((output / "report_manifest.json").read_text())
            self.assertEqual(manifest["plots"]["skipped"], [])
            svg = (output / "plots/mean_mae_across_runs.svg").read_text()
            self.assertIn("AES (MAE; TR / CC / LR / GRA)", svg)
            self.assertIn("0.000 / 0.000 / 0.000 / 0.000", svg)
            self.assertIn("TR", svg)
            self.assertTrue((output / "plots/qwen25_1p5b_calibration_grid.svg").is_file())
            self.assertFalse(plt.get_fignums())

    def test_metric_grids_restore_lines_limits_and_shared_legends(self):
        import matplotlib.pyplot as plt
        from ielts_judge.evaluation.plots import render_plots
        truth, plan, rows = fixture(2)
        plan["models"].append("second")
        rows.extend({**row, "model_key": "second"} for row in list(rows) if row["model_key"] == "qwen25_1p5b")
        tables = report_tables(aligned_results(rows, truth, plan), truth, plan)
        captured = []

        def inspect(fig, path, **settings):
            captured.append(Path(path).stem)
            self.assertEqual(settings["dpi"], 200)
            self.assertEqual(settings["format"], "png")
            self.assertEqual(len(fig.axes), 4)
            self.assertEqual([legend.get_title().get_text() for legend in fig.legends],
                             ["Models", "Prompting configurations"])
            self.assertEqual([text.get_text() for text in fig.legends[1].get_texts()],
                             ["p1: base", "p2: rubric_panda_aes"])
            self.assertIn("TR / CC / LR / GRA", fig.legends[0].get_texts()[-1].get_text())
            for ax in fig.axes:
                self.assertLess(*ax.get_ylim())
                trends = [line for line in ax.lines if line.get_label() in {"Qwen test", "second"}]
                self.assertEqual(len(trends), 2)
                self.assertNotEqual(trends[0].get_color(), trends[1].get_color())
                self.assertTrue(all(line.get_linestyle() == "-" for line in trends))
                self.assertTrue(any(line.get_color() == "red" and line.get_linestyle() == "--" for line in ax.lines))
                self.assertTrue(any(line.get_color() == "black" and line.get_linestyle() == "--" for line in ax.lines))
                separators = [line.get_xdata()[0] for line in ax.lines if line.get_linestyle() == ":"]
                self.assertEqual(separators, [-0.5, 0.5, 1.5])
                self.assertTrue(all(line.get_alpha() == 0.5 for line in ax.get_ygridlines()))
                self.assertEqual(len(ax.get_xticks()), 0)
                stem = Path(path).stem
                if stem == "mean_mae_across_runs":
                    np.testing.assert_allclose(ax.get_ylim(), (-0.01875, 0.39375))
                elif stem == "adjacent_accuracy_across_runs":
                    self.assertEqual(ax.get_ylim(), (0.95, 1.05))
                    self.assertIn("±1 band", ax.get_ylabel())
                elif stem == "mean_qwk_across_runs":
                    self.assertGreater(ax.get_ylim()[1], 1)
                elif "corr" in stem:
                    self.assertGreater(ax.get_ylim()[1], 1)

        with tempfile.TemporaryDirectory() as temp, patch.object(plt.Figure, "savefig", inspect):
            render_plots(tables, plan, Path(temp) / "plots", report_options({"plots": ["metric_comparison"]}))
        self.assertEqual(len(captured), 6)
        self.assertFalse(plt.get_fignums())

    def test_axis_limits_leave_five_percent_visual_margin(self):
        from ielts_judge.evaluation.plots import _axis_limits
        for values, metric, expected in (
            ([2, 4], "mae", (1.9, 4.1)),
            ([0, 9], "mae", (-0.45, 9.45)),
            ([0, 9], "rmse", (-0.45, 9.45)),
            ([0, 81], "mse", (-4.05, 85.05)),
            ([0.6, 0.98], "within_one_accuracy", (0.581, 0.999)),
            ([0.6, 0.8], "within_one_accuracy", (0.59, 0.81)),
            ([-0.8, 0.6], "pearson", (-0.87, 0.67)),
            ([-0.99, 0.99], "spearman", (-1.089, 1.089)),
            ([-0.99, 0.99], "qwk", (-1.089, 1.089)),
            ([-9, 9], "bias", (-9.9, 9.9)),
            ([0, 9], "calibration", (-0.45, 9.45)),
            ([1, 9], "calibration", (0.6, 9.4)),
            ([3, 6], "calibration", (2.85, 6.15)),
            ([0, 0], "mae", (-0.45, 0.45)),
            ([1, 1], "within_one_accuracy", (0.95, 1.05)),
            ([None, np.nan], "pearson", (-1, 1)),
        ):
            with self.subTest(values=values, metric=metric):
                np.testing.assert_allclose(_axis_limits(values, metric), expected)

    def test_grid_and_single_layouts_share_extrema_across_all_criteria(self):
        import matplotlib.pyplot as plt
        from ielts_judge.evaluation.plots import render_plots
        truth, plan, rows = fixture(1)
        for row in rows:
            if row["model_key"] != "aes":
                row["scores"] = {criterion: min(9, row["scores"][criterion] + index)
                                 for index, criterion in enumerate(CRITERIA)}
        tables = report_tables(aligned_results(rows, truth, plan), truth, plan)
        captures = {"grid": {}, "single": {}}
        for layout in captures:
            def inspect(fig, path, **settings):
                captures[layout][Path(path).stem] = [ax.get_ylim() for ax in fig.axes]
            with tempfile.TemporaryDirectory() as temp, patch.object(plt.Figure, "savefig", inspect):
                render_plots(tables, plan, Path(temp) / "plots", report_options({
                    "layout": layout, "metrics": ["mae"], "formats": ["svg"]}))
        self.assertEqual(len(captures["grid"]), 2)
        self.assertEqual(len(captures["single"]), 8)
        for stem, expected in (("mean_mae_across_runs", (-0.1125, 2.3625)),
                               ("qwen25_1p5b_calibration", (-0.45, 9.45))):
            grid_stem = stem + "_grid" if "calibration" in stem else stem
            self.assertEqual(len(captures["grid"][grid_stem]), 4)
            for limits in captures["grid"][grid_stem]:
                np.testing.assert_allclose(limits, expected)
            for criterion in ("TR", "CC", "LR", "GRA"):
                self.assertEqual(len(captures["single"][stem + "_" + criterion]), 1)
                np.testing.assert_allclose(captures["single"][stem + "_" + criterion][0], expected)
        self.assertFalse(plt.get_fignums())

    def test_results_preview_includes_single_calibration_and_excludes_inverse(self):
        from types import ModuleType
        from unittest.mock import Mock
        from PIL import Image
        from ielts_judge.evaluation.results import show_results

        class DisplayImage:
            def __init__(self, filename):
                self.filename = filename

        # Stub notebook presentation only; execute the real file selection and export.
        notebook_display = ModuleType("IPython.display")
        notebook_display.display = Mock()
        notebook_display.Markdown = lambda value: value
        notebook_display.FileLink = lambda value: value
        notebook_display.Image = DisplayImage
        notebook = ModuleType("IPython")
        notebook.display = notebook_display
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            report = run / "reports/one"
            plots = report / "plots"
            plots.mkdir(parents=True)
            (run / "run_manifest.json").write_text(json.dumps({"plan": fixture(1)[1]}))
            (report / "report_manifest.json").write_text("{}")
            for stem in ("qwen25_1p5b_calibration_TR", "qwen25_1p5b_calibration_by_prediction_grid"):
                Image.new("RGB", (1, 1)).save(plots / (stem + ".png"))
            with patch.dict(sys.modules, {"IPython": notebook, "IPython.display": notebook_display}):
                show_results(run, Path(temp) / "downloads")
            previews = [call.args[0] for call in notebook_display.display.call_args_list if isinstance(call.args[0], DisplayImage)]
            self.assertEqual(len(previews), 1)
            self.assertTrue(previews[0].filename.endswith("qwen25_1p5b_calibration_TR.png"))

    def test_calibration_restores_setup_curves_diagonal_and_global_legend(self):
        import matplotlib.pyplot as plt
        from ielts_judge.evaluation.plots import render_plots
        truth, plan, rows = fixture(2)
        for row in rows:
            if row["model_key"] != "aes":
                row["scores"] = {criterion: min(6, max(3, value)) for criterion, value in row["scores"].items()}
        tables = report_tables(aligned_results(rows, truth, plan), truth, plan)

        def inspect(fig, path, **settings):
            self.assertEqual(Path(path).name, "qwen25_1p5b_calibration_grid.png")
            self.assertEqual(len(fig.axes), 4)
            self.assertEqual(len(fig.legends), 1)
            self.assertEqual([text.get_text() for text in fig.legends[0].get_texts()], plan["setups"])
            self.assertEqual(fig.legends[0].get_title().get_text(), "Prompt configurations")
            for ax in fig.axes:
                self.assertEqual(len(ax.containers), len(plan["setups"]))
                np.testing.assert_allclose(ax.get_ylim(), (2.85, 6.15))
                np.testing.assert_array_equal(ax.get_xticks(), range(10))
                np.testing.assert_array_equal(ax.get_yticks(), range(10))
                diagonal = [line for line in ax.lines if line.get_color() == "black"]
                self.assertEqual(len(diagonal), 1)
                self.assertEqual(diagonal[0].get_linestyle(), "--")
                np.testing.assert_array_equal(diagonal[0].get_xdata(), [0, 9])
                np.testing.assert_array_equal(diagonal[0].get_ydata(), [0, 9])
                np.testing.assert_array_equal(ax.containers[0].lines[0].get_xdata(), [0, 3, 6, 9])
                self.assertTrue(ax.get_shared_x_axes().joined(ax, fig.axes[0]))
                self.assertTrue(ax.get_shared_y_axes().joined(ax, fig.axes[0]))
                self.assertTrue(all(line.get_alpha() == 1 for line in ax.get_xgridlines()))

        with tempfile.TemporaryDirectory() as temp, patch.object(plt.Figure, "savefig", inspect):
            render_plots(tables, plan, Path(temp) / "plots", report_options({"plots": ["calibration"]}))
        self.assertFalse(plt.get_fignums())

    def test_nearest_references_exclude_self_and_break_ties_by_id(self):
        vector = np.zeros(768, dtype=np.float32)
        vector[0] = 1
        records = [{"essay_id": essay_id, "essay_hash": essay_id, "task_prompt": "Task", "essay": "Source essay",
                    "insights": "Original insight", "criteria": {c: {"score": 6} for c in CRITERIA}}
                   for essay_id in ("z", "a", "self")]
        references, similarity = nearest_references(np.stack([vector] * 3), records, vector,
                                                     essay_id="self", essay_hash="self", k=2)
        self.assertEqual([r["essay_id"] for r in references], ["a", "z"])
        self.assertEqual(similarity, [1, 1])

    def test_journal_preserves_partial_tail_and_export_is_new(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "segment.jsonl"
            path.write_bytes(b'{"done": 1}\n{"partial":')
            original = path.read_bytes()
            with warnings.catch_warnings(record=True) as caught:
                self.assertEqual(read_segment(path), [{"done": 1}])
            self.assertEqual(len(caught), 1)
            self.assertEqual(path.read_bytes(), original)
            write_new_json(root / "run_manifest.json", {"test": True, "plan": fixture(1)[1]})
            with self.assertRaises(FileExistsError):
                write_new_json(root / "run_manifest.json", {})
            first, second = export_run(root), export_run(root)
            self.assertIn("IELTS_Evaluation_Qwen_test_4_essays_2_prompts_1_repeat", first.name)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_file() and second.is_file())
            restored = restore_export(first, root / "restored")
            self.assertEqual((restored / "segment.jsonl").read_bytes(), original)
            third = export_run(restored)
            with zipfile.ZipFile(third) as archive:
                self.assertEqual(len(archive.namelist()), len(set(archive.namelist())))

    def test_new_kaggle_bootstrap_verifies_pins_and_never_reuses_a_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "inputs/new-evaluation"
            source.mkdir(parents=True)
            archive_path = source / "model_evaluation_bundle.zip"
            with zipfile.ZipFile(archive_path, "x") as archive:
                archive.writestr("code/sample.py", "print('verified')\n")
            import hashlib
            files = {"code/sample.py": hashlib.sha256(b"print('verified')\n").hexdigest()}
            bundle_id = inventory_id(files)
            manifest = {"schema_version": "model-evaluation-bundle-v1", "bundle_id": bundle_id,
                        "archive_name": archive_path.name, "archive_sha256": sha256_file(archive_path), "files": files}
            (source / "ModelEvaluationBundleManifest.json").write_text(json.dumps(manifest))
            first = prepare_project(root / "inputs", root / "work", expected_bundle_id=bundle_id)
            second = prepare_project(root / "inputs", root / "work", expected_bundle_id=bundle_id)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "code/sample.py").read_text(), "print('verified')\n")
            with self.assertRaisesRegex(ValueError, "different bundles"):
                prepare_project(root / "inputs", root / "work", expected_bundle_id="wrong")
            archive_path.unlink()
            (source / "code").mkdir()
            (source / "code/sample.py").write_bytes(b"print('verified')\n")
            unpacked = prepare_project(root / "inputs", root / "work", expected_bundle_id=bundle_id)
            self.assertTrue((unpacked / "code/sample.py").is_file())
            nested = root / "nested-input"
            (nested / "expanded-bundle/code").mkdir(parents=True)
            (nested / "expanded-bundle/code/sample.py").write_bytes(b"print('verified')\n")
            (nested / "ModelEvaluationBundleManifest.json").write_text(json.dumps(manifest))
            nested_project = prepare_project(nested, root / "work", expected_bundle_id=bundle_id)
            self.assertTrue((nested_project / "code/sample.py").is_file())
            (source / "code/sample.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                prepare_project(root / "inputs", root / "work", expected_bundle_id=bundle_id)

    def test_notebook_code_parses_and_has_no_hosted_calls(self):
        import ast
        import importlib.util
        spec = importlib.util.spec_from_file_location("evaluation_bundle_builder", ROOT / "code/tools/build_evaluation_kaggle.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        notebook = module.notebook_document("a" * 64)
        self.assertEqual(json.loads((ROOT / "kaggle/ModelEvaluation.ipynb").read_text(encoding="utf-8")), module.notebook_document())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
        sources = "\n".join(c["source"] for c in notebook["cells"])
        self.assertIn("EXPECTED_BUNDLE_ID = '" + "a" * 64, sources)
        self.assertNotIn("hf.space", sources)
        self.assertIn("EVALUATION_PYTHON", sources)

    def test_incomplete_and_single_repeat_reports_keep_numeric_results(self):
        truth, plan, rows = fixture(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            partial = write_report(rows[:1], truth, plan, root / "partial",
                                   {"plots": ["metric_comparison", "coverage"], "metrics": ["mae"], "formats": ["svg"]})
            manifest = json.loads((partial / "report_manifest.json").read_text())
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["common_valid_count"], 0)
            self.assertEqual(manifest["plots"]["skipped"][0]["type"], "metric_comparison")
            full = write_report(rows, truth, plan, root / "single",
                                {"plots": ["metric_comparison", "score_response"], "metrics": ["mae"], "formats": ["svg"]})
            self.assertTrue((full / "plots/mean_mae_across_runs.svg").is_file())

    def test_plot_failure_closes_figure(self):
        import matplotlib.pyplot as plt
        truth, plan, rows = fixture(1)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                write_report(rows, truth, plan, Path(temp) / "bad-style", {
                    "plots": ["metric_comparison"], "metrics": ["mae"],
                    "style": {"aes": {"color": "this-is-not-a-color"}}})
            self.assertFalse(plt.get_fignums())

    def test_report_subprocess_needs_no_model_weights_or_source_dataset(self):
        import shutil
        import subprocess
        import yaml
        from ielts_judge.data.preparation import file_hash
        truth, plan, rows = fixture(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "minimal-project"
            for source in (ROOT / "code/ielts_judge").rglob("*.py"):
                if "__pycache__" in source.parts:
                    continue
                destination = project / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            for name in ("code/evaluate_models.py", "code/paths.yaml", "code/resources/ielts_criteria.yaml", "kaggle/ModelEvaluationModels.yaml"):
                destination = project / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, destination)
            config = yaml.safe_load((ROOT / "kaggle/ModelEvaluationConfig.yaml").read_text())
            config["report"]["plots"] = []
            config_path = project / "report-only.yaml"
            config_path.write_text(yaml.safe_dump(config))
            run = root / "run"
            run.mkdir()
            truth.assign(dataset_manifest="a" * 64).to_csv(run / "reference.csv", index=False)
            contexts = [{"essay_id": r["essay_id"], "aes": {"scores": r["scores"]}} for r in rows if r["model_key"] == "aes"]
            (run / "contexts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in contexts), encoding="utf-8")
            write_new_json(run / "context_release.json", {"run_fingerprint": "b" * 64, "path": "contexts.jsonl", "sha256": file_hash(run / "contexts.jsonl")})
            write_new_json(run / "run_manifest.json", {"schema_version": "model-evaluation-run-v1", "run_fingerprint": "b" * 64,
                "plan": plan, "reference_sha256": file_hash(run / "reference.csv")})
            (run / "predictions").mkdir()
            (run / "predictions/saved.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["model_key"] != "aes"), encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", str(project / "code/evaluate_models.py"), "--mode", "report",
                                     "--config", str(config_path), "--resume-run", str(run), "--export"],
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(list((run / "reports").glob("*/metrics_summary.csv")))
            self.assertFalse((project / "results/stage3").exists())
            self.assertFalse((project / "code/resources/datasets").exists())

    def test_inference_uses_shared_chain_and_resumes_only_missing_jobs(self):
        config = load_config(ROOT / "kaggle/ModelEvaluationConfig.yaml", ROOT)
        config["setups"] = list(CONTEXT_SETUPS)
        config["runtime"]["judge_device"] = "cpu"
        frame = pd.DataFrame([{"essay_id": f"test:{i}", "essay": f"Essay {i}", "task_prompt": f"Task {i}",
                               "secret_target": "DO_NOT_LEAK_TARGETS"} for i in range(2)])
        contexts = [{"essay_id": r.essay_id, "essay_hash": text_hash(r.essay),
                     "aes": {"scores": dict.fromkeys(CRITERIA, 6), "input": {}, "semantics": "essay-only"},
                     "references": [{"essay_id": "train:1", "task_prompt": "Other task", "essay": "Other essay",
                                     "scores": dict.fromkeys(CRITERIA, 5), "insights": "Original source insight"}],
                     "retrieval": {"status": "ok"}} for r in frame.itertuples()]
        calls = []

        class JudgeDouble:
            def __init__(self, key, registry):
                self.registry, self.profile = registry, registry.require_enabled(key)

            def count_input_tokens(self, messages):
                return 100

            def predict_messages(self, *, model_key, messages, deadline):
                calls.append(messages)
                result = {"raw_text": json.dumps(dict.fromkeys(CRITERIA, 6)), "model_id": self.profile.repo_id,
                          "model_revision": self.profile.revision, "prompt_version": PROMPT_VERSION,
                          "output_schema_version": OUTPUT_SCHEMA_VERSION, "input_tokens": 100,
                          "generated_tokens": 40, "max_new_tokens": 256, "stop_reason": "eos"}
                self.last_generation = result
                return result

        manifest = {"run_fingerprint": "b" * 64, "plan": {"dataset_manifest_sha256": "a" * 64}}
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            self.assertEqual(infer_models(ROOT, config, run, frame, contexts, manifest, max_jobs=3, judge_factory=JudgeDouble), 3)
            self.assertEqual(infer_models(ROOT, config, run, frame, contexts, manifest, judge_factory=JudgeDouble), 9)
            self.assertEqual(infer_models(ROOT, config, run, frame, contexts, manifest, judge_factory=JudgeDouble), 0)
            results = prediction_records(run)
            self.assertEqual(len(results), 12)
            self.assertTrue(all(r["status"] == "ok" for r in results))
            self.assertTrue(all(r["metadata"]["judge_attempts"] == 1 for r in results))
            self.assertEqual(len(calls), 12)
            self.assertNotIn("DO_NOT_LEAK_TARGETS", json.dumps(calls))
            self.assertIn("Original source insight", json.dumps(calls))
            class InvalidJudge(JudgeDouble):
                def predict_messages(self, **kwargs):
                    result = super().predict_messages(**kwargs)
                    result["raw_text"] = "invalid completion"
                    self.last_generation = result
                    return result
            failed_run = run / "invalid-output-run"
            failed_run.mkdir()
            self.assertEqual(infer_models(ROOT, config, failed_run, frame, contexts, manifest,
                                         max_jobs=1, judge_factory=InvalidJudge), 1)
            failure = prediction_records(failed_run)[0]
            self.assertEqual(failure["error"]["code"], "invalid_output")
            self.assertEqual(failure["raw_completion"], "invalid completion")
            self.assertEqual(failure["metadata"]["generation"]["generated_tokens"], 40)


if __name__ == "__main__":
    unittest.main()
