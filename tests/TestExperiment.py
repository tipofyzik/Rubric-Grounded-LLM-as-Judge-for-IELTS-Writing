"""Experiment selection, persisted replay, consumer routing and corpus exclusions."""
from typing import Any

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]

from offline_research import PrepareData as runner
from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from ielts_judge.data.Experiment import evaluation_options, prepare_experiment, validate_reference_corpus
from ielts_judge.data.Preparation import file_hash, text_hash
from ielts_judge.Paths import PATHS


def rows(split: Any, count: int) -> Any:
    """Rows.

    Args:
        split (Any): Split used by this operation.
        count (int): Count used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    frame = pd.DataFrame([
        {"essay_id": f"writing9:{split}:{current_index}", "task_prompt": f"Task {current_index}",
         "essay": f"Unique {split} essay {current_index}", "essay_hash": text_hash(f"Unique {split} essay {current_index}"),
         "dataset_source": "writing9", "status": "accepted", "split": split,
         **{name: 4 + current_index % 5 for name in PUBLIC_CRITERION_IDS}}
        for current_index in range(count)])
    frame.attrs["manifest_sha256"] = "a" * 64
    return frame


class ExperimentTests(unittest.TestCase):
    """Manage experiment tests through the operations defined by this component."""
    def setUp(self) -> None:
        """SetUp.

        Returns:
            None: This operation returns no value.
        """
        self.train, self.pool = rows("train", 8), rows("test", 12)

    def test_full_train_and_pool_are_unchanged_and_questions_may_overlap(self) -> None:
        """Verify that full train and pool are unchanged and questions may overlap.

        Returns:
            None: This operation returns no value.
        """
        before_train, before_pool = self.train.copy(deep=True), self.pool.copy(deep=True)
        selected, manifest = prepare_experiment(self.train, self.pool)
        pd.testing.assert_frame_equal(self.train, before_train)
        pd.testing.assert_frame_equal(self.pool, before_pool)
        self.assertEqual(set(selected.essay_id), set(self.pool.essay_id))
        self.assertEqual(manifest["reference_corpus"]["count"], len(self.train))
        self.assertEqual(manifest["reference_corpus"]["records_sha256"], manifest["train"]["records_sha256"])
        self.assertEqual(manifest["question_policy"]["shared_train_test_questions"], 8)
        self.assertEqual(manifest["protocol"]["aes_training_provenance"], "unknown")

    def test_sampling_is_order_independent_seeded_and_nested(self) -> None:
        """Verify that sampling is order independent seeded and nested.

        Returns:
            None: This operation returns no value.
        """
        selected, manifest = prepare_experiment(self.train, self.pool, config={"sample_size": 4, "seed": 5})
        reordered, repeated = prepare_experiment(self.train.iloc[::-1], self.pool.iloc[::-1],
                                                 config={"sample_size": 4, "seed": 5})
        self.assertEqual(manifest, repeated)
        pd.testing.assert_frame_equal(selected, reordered)
        changed, _ = prepare_experiment(self.train, self.pool, config={"sample_size": 4, "seed": 6})
        self.assertNotEqual(set(selected.essay_id), set(changed.essay_id))
        larger, _ = prepare_experiment(self.train, self.pool, config={"sample_size": 6, "seed": 5})
        self.assertTrue(set(selected.essay_id).issubset(set(larger.essay_id)))

    def test_invalid_options_and_oversized_request_are_rejected(self) -> None:
        """Verify that invalid options and oversized request are rejected.

        Returns:
            None: This operation returns no value.
        """
        for config in ({"sample_size": 0}, {"sample_size": True}, {"sample_size": 1.5},
                       {"seed": -1}, {"seed": False}, {"typo": 4}, [], False):
            with self.subTest(config=config), self.assertRaises(ValueError):
                evaluation_options(config)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            prepare_experiment(self.train, self.pool, config={"sample_size": 13})

    def test_persisted_manifest_replays_and_rejects_tampering_or_changed_data(self) -> None:
        """Verify that persisted manifest replays and rejects tampering or changed data.

        Returns:
            None: This operation returns no value.
        """
        selected, manifest = prepare_experiment(self.train, self.pool, config={"sample_size": 3})
        saved = json.loads(json.dumps(manifest))
        replayed, _ = prepare_experiment(self.train, self.pool.iloc[::-1], saved_manifest=saved,
                                         config={"sample_size": 1})
        pd.testing.assert_frame_equal(selected, replayed)
        for key in ("dataset_manifest_sha256", "dataset_source", "schema_version"):
            broken = deepcopy(saved)
            broken[key] = "wrong"
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_experiment(self.train, self.pool, saved_manifest=broken)
        broken = deepcopy(saved)
        broken["evaluation"]["essay_ids"][0] = self.train.iloc[0].essay_id
        with self.assertRaises(ValueError):
            prepare_experiment(self.train, self.pool, saved_manifest=broken)
        altered = self.pool.copy()
        altered.loc[0, PUBLIC_CRITERION_IDS[0]] = 9
        with self.assertRaises(ValueError):
            prepare_experiment(self.train, altered, saved_manifest=saved)

    def test_missing_task_stale_text_hash_and_mixed_releases_are_rejected(self) -> None:
        """Verify that missing task stale text hash and mixed releases are rejected.

        Returns:
            None: This operation returns no value.
        """
        for column, value in (("task_prompt", ""), ("essay", "Changed text"), ("essay_id", "")):
            broken = self.pool.copy()
            broken.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                prepare_experiment(self.train, broken)
        broken = self.pool.copy()
        broken.attrs["manifest_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "same verified"):
            prepare_experiment(self.train, broken)

    def test_full_corpus_rejects_subsets_test_rows_changed_labels_and_duplicates(self) -> None:
        """Verify that full corpus rejects subsets test rows changed labels and duplicates.

        Returns:
            None: This operation returns no value.
        """
        _, manifest = prepare_experiment(self.train, self.pool)
        validate_reference_corpus(self.train.iloc[::-1], manifest)
        fractional = self.train.astype({PUBLIC_CRITERION_IDS[0]: float})
        fractional.loc[0, PUBLIC_CRITERION_IDS[0]] = 4.5
        altered = self.train.copy()
        altered.loc[0, PUBLIC_CRITERION_IDS[0]] = 9
        duplicate = pd.concat([self.train, self.train.iloc[:1]], ignore_index=True)
        duplicate.attrs = dict(self.train.attrs)
        for invalid in (self.train.iloc[:-1], self.pool, fractional, altered, duplicate):
            with self.subTest(count=len(invalid)), self.assertRaises(ValueError):
                validate_reference_corpus(invalid, manifest)

    def test_same_text_under_different_ids_cannot_enter_test(self) -> None:
        """Verify that same text under different ids cannot enter test.

        Returns:
            None: This operation returns no value.
        """
        broken = self.pool.copy()
        broken.loc[0, ["essay", "essay_hash"]] = self.train.iloc[0][["essay", "essay_hash"]].to_numpy()
        with self.assertRaisesRegex(ValueError, "leakage"):
            prepare_experiment(self.train, broken)

    def test_corpus_descriptor_cannot_redefine_full_train_as_a_subset(self) -> None:
        """Verify that corpus descriptor cannot redefine full train as a subset.

        Returns:
            None: This operation returns no value.
        """
        _, manifest = prepare_experiment(self.train, self.pool)
        _, small = prepare_experiment(self.train.iloc[:-1], self.pool)
        broken = deepcopy(manifest)
        broken["reference_corpus"] = small["reference_corpus"]
        with self.assertRaisesRegex(ValueError, "descriptor"):
            validate_reference_corpus(self.train.iloc[:-1], broken)

    def test_retrieval_checks_full_membership_before_embedding_and_excludes_copied_query(self) -> None:
        """Verify that retrieval checks full membership before embedding and excludes copied query.

        Returns:
            None: This operation returns no value.
        """
        from offline_research.panda.RetrievalEvaluator import PandaRetrievalEvaluator

        _, manifest = prepare_experiment(self.train, self.pool)
        retrieval = PandaRetrievalEvaluator("")
        self.enterContext(patch.object(
            retrieval, "embedder", SimpleNamespace(encode=lambda texts, **keyword_arguments: np.ones((len(texts), 2))),
            create=True))
        retrieval.panda_insights = pd.DataFrame({
            "essay_id": self.train.essay_id.iloc[::-1].tolist(),
            "dataset_manifest": ["a" * 64] * len(self.train),
            "insight_1": ["A source-score explanation"] * len(self.train),
        })
        for invalid in (self.train.iloc[:-1], self.pool, self.train.essay.to_numpy()):
            with self.subTest(count=len(invalid)), self.assertRaises(ValueError):
                retrieval.build_retrieval_index(invalid, experiment_manifest=manifest)
        retrieval.build_retrieval_index(self.train.iloc[::-1], experiment_manifest=manifest)
        _, references = retrieval.find_similar_essays(
            {"essay_id": "different-source-id", "essay": self.train.iloc[0].essay}, num_similar=20)
        self.assertEqual(len(references), len(self.train) - 1)
        self.assertNotIn(self.train.iloc[0].essay_id, [record.essay_id for record in references])


class MainExperimentTests(unittest.TestCase):
    """Manage main experiment tests through the operations defined by this component."""
    def setUp(self) -> None:
        """SetUp.

        Returns:
            None: This operation returns no value.
        """
        (ROOT / "TEMP").mkdir(exist_ok=True)
        folder = tempfile.TemporaryDirectory(dir=ROOT / "TEMP")
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.train, self.pool = rows("train", 8), rows("test", 12)
        self.config_path = self.root / "config.yaml"
        self.config = {"tasks": {}, "dataset": {"source": "combined"},
                       "evaluation": {"sample_size": 3, "seed": 42}, "inference": {"runs": 1}}
        self.paths = deepcopy(PATHS)
        self.paths["results"]["dataset_runs"] = str(self.root / "runs")
        for context in (patch.dict(PATHS, self.paths), patch.object(runner, "CONFIG_PATH", self.config_path),
                        patch.object(runner, "load_data", return_value=(self.train, self.pool, {}))):
            context.start()
            self.addCleanup(context.stop)

    def run_main(self, **keyword_arguments: Any) -> Any:
        """Run main.

        Args:
            keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            return runner.main(**keyword_arguments)

    def manifests(self) -> Any:
        """Manifests.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        return sorted((self.root / "runs").rglob("experiment_manifest.json"))

    def test_preparation_never_runs_models_and_runs_are_separate(self) -> None:
        """Verify that preparation never runs models and runs are separate.

        Returns:
            None: This operation returns no value.
        """
        self.config["tasks"] = {"train_custom_regressor": 1, "panda_evaluation": 1}
        train, selected, _ = self.run_main(prepare_only=True)
        self.assertEqual((len(train), len(selected)), (8, 3))
        first = self.manifests()[0]
        content = first.read_bytes()
        self.run_main(prepare_only=True)
        self.assertEqual(len(self.manifests()), 2)
        self.assertEqual(first.read_bytes(), content)

    def test_replay_and_resume_keep_selection_even_if_config_changes(self) -> None:
        """Verify that replay and resume keep selection even if config changes.

        Returns:
            None: This operation returns no value.
        """
        _, first_test, _ = self.run_main()
        saved = self.manifests()[0]
        saved_hash = file_hash(saved)
        self.config["evaluation"] = {"sample_size": 1, "seed": 99}
        _, replayed, _ = self.run_main(selection_manifest=saved)
        pd.testing.assert_frame_equal(first_test, replayed)
        count = len(self.manifests())
        _, resumed, _ = self.run_main(run_directory=saved.parent)
        pd.testing.assert_frame_equal(first_test, resumed)
        self.assertEqual(len(self.manifests()), count)
        self.assertEqual(file_hash(saved), saved_hash)
        with self.assertRaises(ValueError):
            self.run_main(selection_manifest=saved, run_directory=saved.parent)

    def test_all_model_consumers_receive_full_train_and_the_same_selected_test(self) -> None:
        """Verify that all model consumers receive full train and the same selected test.

        Returns:
            None: This operation returns no value.
        """
        calls = {}

        def aes_run(train: Any, test: Any, experiment: Any, result_root: Any, **keyword_arguments: Any) -> None:
            # This test covers main's routing. TestAES runs the real AES pipeline.
            """Aes run.

            Args:
                train (Any): Train used by this operation.
                test (Any): Test used by this operation.
                experiment (Any): Experiment used by this operation.
                result_root (Any): Result root used by this operation.
                keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

            Returns:
                None: This operation returns no value.
            """
            calls["aes_train"] = train.essay.tolist()
            calls["aes_test"] = test.essay.tolist()
            test[["essay_id"]].to_csv(result_root / "aes_predictions.csv", index=False)

        class SFT:
            """Manage sft through the operations defined by this component."""
            def __init__(self, **keyword_arguments: Any) -> None:
                """Initialize this component's settings and collaborators.

                Args:
                    keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

                Returns:
                    None: This operation returns no value.
                """
                pass

            def train(self, records: Any, labels: Any) -> None:
                """Train.

                Args:
                    records (Any): Records used by this operation.
                    labels (Any): Labels used by this operation.

                Returns:
                    None: This operation returns no value.
                """
                calls["sft_train"] = records

            def evaluate_llm(self, records: Any, labels: Any, **keyword_arguments: Any) -> None:
                """Evaluate llm.

                Args:
                    records (Any): Records used by this operation.
                    labels (Any): Labels used by this operation.
                    keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

                Returns:
                    None: This operation returns no value.
                """
                calls["sft_test"] = records

        class Obtainer:
            """Manage obtainer through the operations defined by this component."""
            def __init__(self, **keyword_arguments: Any) -> None:
                """Initialize this component's settings and collaborators.

                Args:
                    keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

                Returns:
                    None: This operation returns no value.
                """
                pass

            def load_model_and_tokenizer(self, **keyword_arguments: Any) -> None:
                """Load model and tokenizer.

                Args:
                    keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

                Returns:
                    None: This operation returns no value.
                """
                pass

            def get_response_locally(self, number_of_evaluations: Any, test_dataset: Any) -> None:
                """Get response locally.

                Args:
                    number_of_evaluations (Any): Number of evaluations used by this operation.
                    test_dataset (Any): Test dataset used by this operation.

                Returns:
                    None: This operation returns no value.
                """
                calls["inference_test"] = test_dataset

        class Panda(Obtainer):
            """Manage panda through the operations defined by this component."""
            def retrieve_insights(self, essays: Any, scores: Any, submissions: Any) -> None:
                """Retrieve insights.

                Args:
                    essays (Any): Essays used by this operation.
                    scores (Any): Four criterion scores using the public IELTS score contract.
                    submissions (Any): Submissions used by this operation.

                Returns:
                    None: This operation returns no value.
                """
                calls["panda_train"] = submissions

        modules = {
            "offline_research.aes.Training": SimpleNamespace(run_aes=aes_run),
            "offline_research.llm.LLMFineTuner": SimpleNamespace(LLMFineTuner=SFT),
            "offline_research.llm.LLMResponseObtainer": SimpleNamespace(LLMResponseObtainer=Obtainer),
            "offline_research.panda.Panda": SimpleNamespace(PANDA=Panda),
        }
        self.config["tasks"] = dict.fromkeys(
            ("train_custom_regressor", "fine_tune_llm", "get_llm_predictions", "panda_evaluation"), 1)
        with ExitStack() as stack:
            for name, replacement in modules.items():
                module = importlib.import_module(name)
                for attribute, value in vars(replacement).items():
                    stack.enter_context(patch.object(module, attribute, value))
            train, selected, _ = self.run_main()
        self.assertEqual(calls["aes_train"], train.essay.tolist())
        for key in ("sft_train", "panda_train"):
            self.assertEqual({current_record["essay_id"] for current_record in calls[key]}, set(train.essay_id))
            self.assertTrue(all(set(current_record).isdisjoint(PUBLIC_CRITERION_IDS) for current_record in calls[key]))
        self.assertEqual(calls["aes_test"], selected.essay.tolist())
        self.assertEqual({current_record["essay_id"] for current_record in calls["sft_test"]}, set(selected.essay_id))
        pd.testing.assert_frame_equal(calls["inference_test"], selected)
        result = self.manifests()[0].parent
        truth = pd.read_csv(result / "y_test.csv")
        self.assertEqual(set(truth["index"]), set(selected.essay_id))
        self.assertEqual(len(pd.read_parquet(result / "train_llm.parquet")), len(train))
        self.assertEqual(len(pd.read_parquet(result / "test_llm.parquet")), len(selected))
        self.assertEqual(set(pd.read_csv(result / "aes_predictions.csv").essay_id), set(selected.essay_id))


if __name__ == "__main__":
    unittest.main()
