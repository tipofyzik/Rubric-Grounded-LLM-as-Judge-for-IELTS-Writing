"""Task and ID propagation through inference, SFT, retrieval and evaluation."""
from typing import Any

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from ielts_judge.Criteria import CRITERION_IDS, PUBLIC_CRITERION_IDS
from ielts_judge.data.Preparation import text_hash
from offline_research.data.Runtime import convert_to_llm_dataset, export_reference, submission_records
from offline_research.evaluation.LLMAggregator import LLMAggregator
from offline_research.evaluation.LLMEvaluator import LLMEvaluator
from offline_research.llm.LLMResponseObtainer import LLMResponseObtainer
from offline_research.llm.LLMFineTuner import LLMFineTuner
from ielts_judge.llm.Generation import GenerationResult
from offline_research.panda.InsightGenerator import PandaInsightGenerator
from offline_research.panda.RetrievalEvaluator import PandaRetrievalEvaluator
from ielts_judge.Paths import PATHS


class DatasetConsumerTests(unittest.TestCase):
    """Manage dataset consumer tests through the operations defined by this component."""
    def setUp(self) -> None:
        """SetUp.

        Returns:
            None: This operation returns no value.
        """
        (ROOT/'TEMP').mkdir(exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(dir=ROOT/'TEMP')
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.frame = pd.DataFrame([
            {'essay_id': f'writing9:train:{current_index}', 'task_prompt': f'Task {current_index}', 'essay': f'Essay {current_index}',
             'essay_hash': text_hash(f'Essay {current_index}'), 'status': 'accepted', 'split': 'train',
             **dict.fromkeys(PUBLIC_CRITERION_IDS, current_index + 5), 'forbidden_gt': 'DO_NOT_LEAK_TARGETS'}
            for current_index in range(3)])
        self.frame.attrs['manifest_sha256'] = 'a'*64

    def test_test_prompts_contain_task_but_no_target_and_training_targets_are_explicit(self) -> None:
        """Verify that test prompts contain task but no target and training targets are explicit.

        Returns:
            None: This operation returns no value.
        """
        testing = convert_to_llm_dataset(self.frame, test_data=True)
        training = convert_to_llm_dataset(self.frame)
        self.assertEqual(testing.essay_id.tolist(), self.frame.essay_id.tolist())
        self.assertIn('Task 0', testing.iloc[0].text)
        self.assertTrue(testing.iloc[0].text.endswith('### Answer:\n'))
        self.assertNotIn('DO_NOT_LEAK_TARGETS', testing.iloc[0].text)
        self.assertEqual(json.loads(testing.iloc[0].evaluation)[CRITERION_IDS[0]], 5)
        self.assertIn('"task_achievement": 5', training.iloc[0].text)
        tuner = object.__new__(LLMFineTuner)
        prompt = tuner._build_prompt(submission_records(self.frame)[0], [5]*4)
        self.assertIn('Task 0', prompt)
        self.assertIn('Essay 0', prompt)

    def test_inference_uses_real_record_adapter_and_persists_ids(self) -> None:
        """Verify that inference uses real record adapter and persists ids.

        Returns:
            None: This operation returns no value.
        """
        answer = dict.fromkeys(CRITERION_IDS, 6)
        generated = GenerationResult(json.dumps(answer), 30, 12, 200, 2, 'eos')
        obtainer = LLMResponseObtainer('')
        with patch.object(obtainer, 'model', object(), create=True), \
             patch.object(obtainer, 'tokenizer', object(), create=True), \
             patch.dict(PATHS['results'], llm_hugging_face_responses=str(self.path)), \
             patch('offline_research.llm.LLMResponseObtainer.generate_response', return_value=generated) as generator, \
             redirect_stdout(io.StringIO()):
            obtainer.get_response_locally(1, self.frame.iloc[::-1])
        messages = generator.call_args_list[0].args[2]
        self.assertIn('Task 2', messages[1]['content'])
        self.assertNotIn('DO_NOT_LEAK_TARGETS', messages[1]['content'])
        results = json.loads((self.path/PATHS['files']['llm_response'].format(evaluation=1)).read_text())
        self.assertEqual([current_value['essay_id'] for current_value in results], self.frame.essay_id.tolist()[::-1])
        self.assertTrue(all(current_value['dataset_manifest'] == 'a'*64 for current_value in results))

    def test_aggregation_and_metric_alignment_follow_ids_after_reordering(self) -> None:
        """Verify that aggregation and metric alignment follow ids after reordering.

        Returns:
            None: This operation returns no value.
        """
        export_reference(self.frame, self.path/'truth.csv')
        records = [{'essay_id': row.essay_id, 'dataset_manifest': 'a'*64,
                    'answer': {name: int(row[name]) for name in PUBLIC_CRITERION_IDS}}
                   for _, row in self.frame.iloc[::-1].iterrows()]
        raw = self.path/'raw'
        raw.mkdir()
        (raw/'qwen25_1.json').write_text(json.dumps(records))
        LLMAggregator(str(raw)).run(str(self.path/'predictions.csv'))
        predictions = pd.read_csv(self.path/'predictions.csv')
        evaluator = LLMEvaluator(str(self.path), str(self.path/'truth.csv'))
        true, current_prediction = evaluator._get_true_pred(predictions.sample(frac=1), CRITERION_IDS[0])
        np.testing.assert_array_equal(true, current_prediction)
        with self.assertRaisesRegex(ValueError, 'different dataset'):
            evaluator._get_true_pred(predictions.assign(dataset_manifest='wrong'), CRITERION_IDS[0])
        records.append(records[0])
        (raw/'qwen25_1.json').write_text(json.dumps(records))
        with self.assertRaisesRegex(ValueError, 'Duplicate essay'):
            LLMAggregator(str(raw)).run(str(self.path/'bad.csv'))

    def test_failed_prediction_is_not_replaced_with_another_essays_score(self) -> None:
        """Verify that failed prediction is not replaced with another essays score.

        Returns:
            None: This operation returns no value.
        """
        records = [{'essay_id': row.essay_id, 'dataset_manifest': 'a'*64, 'answer': None}
                   for _, row in self.frame.iterrows()]
        (self.path/'qwen25_1.json').write_text(json.dumps(records))
        LLMAggregator(str(self.path)).run(str(self.path/'results.csv'))
        result = pd.read_csv(self.path/'results.csv')
        self.assertTrue(result[list(CRITERION_IDS)].isna().all().all())

    def retrieval(self) -> Any:
        """Retrieval.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        class Embedder:
            """Manage embedder through the operations defined by this component."""
            def encode(self, texts: Any, **keyword_arguments: Any) -> Any:
                """Encode.

                Args:
                    texts (Any): Texts used by this operation.
                    keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

                Returns:
                    Any: The computed result for the supplied inputs.
                """
                return np.array([[1.0, float(len(value))] for value in texts])
        retrieval = PandaRetrievalEvaluator('')
        embedder_patch = patch.object(retrieval, 'embedder', Embedder(), create=True)
        embedder_patch.start()
        self.addCleanup(embedder_patch.stop)
        retrieval.panda_insights = pd.DataFrame([
            {'essay_id': row.essay_id, 'insight_1': f'insight for {row.essay_id}', 'dataset_manifest': 'a'*64}
            for _,row in self.frame.iloc[::-1].iterrows()])
        return retrieval

    def test_retrieval_aligns_insights_by_id_and_excludes_query(self) -> None:
        """Verify that retrieval aligns insights by id and excludes query.

        Returns:
            None: This operation returns no value.
        """
        retrieval = self.retrieval()
        retrieval.build_retrieval_index(self.frame)
        self.assertEqual(retrieval.panda_insights.essay_id.tolist(), self.frame.essay_id.tolist())
        _, insights = retrieval.find_similar_essays(submission_records(self.frame)[0], num_similar=3)
        self.assertEqual(len(insights), 2)
        self.assertNotIn(self.frame.iloc[0].essay_id, [row.essay_id for row in insights])
        messages = retrieval.build_evaluation_messages(submission_records(self.frame)[0], insights)
        self.assertIn('Task 0', messages[1]['content'])

    def test_retrieval_rejects_test_rows_and_stale_insight_release(self) -> None:
        """Verify that retrieval rejects test rows and stale insight release.

        Returns:
            None: This operation returns no value.
        """
        with self.assertRaisesRegex(ValueError, 'train rows'):
            self.retrieval().build_retrieval_index(self.frame.assign(split='test'))
        retrieval = self.retrieval()
        retrieval.panda_insights['dataset_manifest'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'different dataset'):
            retrieval.build_retrieval_index(self.frame)

    def test_panda_zero_is_not_relabelled_as_one_and_includes_original_task(self) -> None:
        """Verify that panda zero is not relabelled as one and includes original task.

        Returns:
            None: This operation returns no value.
        """
        prompt = PandaInsightGenerator('').build_insight_prompt('Essay', np.array([0, 1, 8, 9]), 'Task about schools')
        self.assertIn('Task about schools', prompt)
        self.assertIn('Task Response: 0', prompt)
        self.assertIn('source score of 0 is retained as 0', prompt)

    def test_aes_encoding_supports_all_ten_integer_classes_without_loading_model(self) -> None:
        """Verify that aes encoding supports all ten integer classes without loading model.

        Returns:
            None: This operation returns no value.
        """
        from ielts_judge.aes.OrdinalRegression import OrdinalRegression
        labels = pd.DataFrame([[0]*4, [9]*4], columns=CRITERION_IDS)
        encoded = OrdinalRegression.ordinal_encode_multi(labels, {current_index:current_index for current_index in range(10)}, 10)
        self.assertEqual(encoded.shape, (2, 4, 9))
        self.assertTrue((encoded[0] == 0).all())
        self.assertTrue((encoded[1] == 1).all())


if __name__ == '__main__':
    unittest.main()
