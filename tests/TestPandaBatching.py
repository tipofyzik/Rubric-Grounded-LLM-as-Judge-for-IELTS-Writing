"""Batch boundaries, recovery, left padding and real local Transformers generation."""

from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]

from TestExperiment import rows
from TestPandaCorpus import batch_generator, explanation
from ielts_judge.data.Experiment import prepare_experiment
from ielts_judge.panda.Artifacts import canonical, read_json
from offline_research.panda.Contracts import generation_profile, result_path, row_input
from offline_research.panda.Generation import HFInsightGenerator, pilot_order, run_insights


def tiny_tokenizer() -> Any:
    """Tiny tokenizer.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "short": 2, "long": 3,
                                  "medium": 4, "answer": 5, "other": 6, "word": 7}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                        eos_token="[EOS]", pad_token="[EOS]", padding_side="left")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    return tokenizer


class PandaBatchingTests(unittest.TestCase):
    """Manage panda batching tests through the operations defined by this component."""
    def setUp(self) -> None:
        """SetUp.

        Returns:
            None: This operation returns no value.
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.train, self.test = rows("train", 7), rows("test", 3)
        _, self.experiment = prepare_experiment(self.train, self.test)
        self.profile = generation_profile({"batch_size": 3})

    def test_batch_size_is_a_positive_integer_and_part_of_profile(self) -> None:
        """Verify that batch size is a positive integer and part of profile.

        Returns:
            None: This operation returns no value.
        """
        self.assertEqual(generation_profile()["batch_size"], 12)
        for invalid in (0, -1, True, 2.5, "4", None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "batch_size"):
                generation_profile({"batch_size": invalid})

    def test_limit_counts_essays_partial_batch_and_resume_preserve_ids(self) -> None:
        """Verify that limit counts essays partial batch and resume preserve ids.

        Returns:
            None: This operation returns no value.
        """
        calls = []
        def generate(records: Any) -> list[Any]:
            """Generate.

            Args:
                records (Any): Records used by this operation.

            Returns:
                list[Any]: The computed result for the supplied inputs.
            """
            calls.append([record["essay_id"] for record in records])
            return [(canonical(explanation(record)), {"stop_reason": "eos"}) for record in records]
        output = self.path / "run"
        report = run_insights(self.train, self.experiment, output, self.profile, generate, limit=5)
        self.assertEqual([len(batch) for batch in calls], [3, 2])
        self.assertEqual((report["accepted"], report["pending"], report["attempted_this_invocation"]), (5, 2, 5))
        self.assertGreater(report["processing_seconds_this_invocation"], 0)
        self.assertAlmostEqual(report["seconds_per_attempt_this_invocation"], report["processing_seconds_this_invocation"] / 5)
        report = run_insights(self.train.iloc[::-1], self.experiment, output, self.profile, generate)
        self.assertEqual([len(batch) for batch in calls], [3, 2, 2])
        self.assertEqual(len({essay_id for batch in calls for essay_id in batch}), 7)
        self.assertTrue(report["complete"])
        for row in self.train.to_dict("records"):
            result = read_json(result_path(output, row["essay_id"]))
            self.assertEqual(result["essay_id"], row["essay_id"])
            self.assertEqual(result["insights"], explanation(row_input(row)))
        report = run_insights(self.train, self.experiment, output, self.profile, generate)
        self.assertEqual(len(calls), 3)
        self.assertIsNone(report["seconds_per_attempt_this_invocation"])
        with self.assertRaisesRegex(ValueError, "profile/train"):
            run_insights(self.train, self.experiment, output, generation_profile({"batch_size": 2}), generate)

    def test_individual_budget_and_format_errors_do_not_discard_other_answers(self) -> None:
        """Verify that individual budget and format errors do not discard other answers.

        Returns:
            None: This operation returns no value.
        """
        ordered = pilot_order(self.train)
        rejected = ordered[1]["essay_id"]
        bad_format = ordered[2]["essay_id"]
        def generate(records: Any) -> list[Any]:
            """Generate.

            Args:
                records (Any): Records used by this operation.

            Returns:
                list[Any]: The computed result for the supplied inputs.
            """
            return [ValueError("Input token budget exceeded") if record["essay_id"] == rejected else
                    ('{"unfinished":', {"stop_reason": "eos"}) if record["essay_id"] == bad_format else
                    (canonical(explanation(record)), {"stop_reason": "eos"}) for record in records]
        output = self.path / "run"
        report = run_insights(self.train, self.experiment, output, self.profile, generate)
        self.assertEqual((report["accepted"], report["failed"]), (5, 2))
        result = read_json(result_path(output, rejected))
        self.assertEqual(result["error"], "Input token budget exceeded")
        self.assertIsNone(result["insights"])
        self.assertNotIn("completion", result)
        result = read_json(result_path(output, bad_format))
        self.assertEqual(result["completion"], '{"unfinished":')
        self.assertIsNone(result["insights"])
        calls = []
        def repaired(records: Any) -> list[Any]:
            """Repaired.

            Args:
                records (Any): Records used by this operation.

            Returns:
                list[Any]: The computed result for the supplied inputs.
            """
            calls.extend(record["essay_id"] for record in records)
            return [(canonical(explanation(record)), {"stop_reason": "eos"}) for record in records]
        run_insights(self.train, self.experiment, output, self.profile, repaired)
        self.assertEqual(calls, [])
        report = run_insights(self.train, self.experiment, output, self.profile, repaired, retry_failed=True)
        self.assertEqual(set(calls), {rejected, bad_format})
        self.assertTrue(report["complete"])

    def test_interrupted_batch_preserves_previous_batch_and_resumes(self) -> None:
        """Verify that interrupted batch preserves previous batch and resumes.

        Returns:
            None: This operation returns no value.
        """
        output = self.path / "run"
        calls = []
        def generate(records: Any) -> list[Any]:
            """Generate.

            Args:
                records (Any): Records used by this operation.

            Returns:
                list[Any]: The computed result for the supplied inputs.
            """
            calls.append([record["essay_id"] for record in records])
            if len(calls) == 2:
                raise RuntimeError("CUDA out of memory")
            return [(canonical(explanation(record)), {"stop_reason": "eos"}) for record in records]
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            run_insights(self.train, self.experiment, output, self.profile, generate)
        self.assertEqual(len(list((output / "responses").glob("*.json"))), 3)
        report = run_insights(self.train, self.experiment, output, self.profile,
                              batch_generator(lambda current_record: (canonical(explanation(current_record)), {"stop_reason": "eos"})))
        self.assertEqual(report["attempted_this_invocation"], 4)
        self.assertTrue(report["complete"])

    def test_wrong_batch_result_count_does_not_write_misaligned_responses(self) -> None:
        """Verify that wrong batch result count does not write misaligned responses.

        Returns:
            None: This operation returns no value.
        """
        output = self.path / "run"
        with self.assertRaisesRegex(ValueError, "number of batch results"):
            run_insights(self.train, self.experiment, output, self.profile, lambda records: [])
        self.assertFalse((output / "responses").exists())

    def test_real_tokenizer_left_padding_early_eos_and_budget_isolation(self) -> None:
        """Verify that real tokenizer left padding early eos and budget isolation.

        Returns:
            None: This operation returns no value.
        """
        import torch
        tokenizer = tiny_tokenizer()
        captured = []
        def generate(**keyword_arguments: Any) -> Any:
            """Generate.

            Args:
                keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            captured.append(keyword_arguments)
            suffix = torch.tensor([[5, 1, 1, 1], [5, 6, 1, 1], [5, 6, 6, 6]])
            return torch.cat((keyword_arguments["input_ids"], suffix), dim=1)
        runtime: Any = SimpleNamespace(tokenizer=tokenizer,
            build_insight_prompt=lambda essay, *current_arguments, **keyword_arguments: essay,
            model=SimpleNamespace(device="cpu", config=SimpleNamespace(max_position_embeddings=32),
                                  generation_config=SimpleNamespace(eos_token_id=[1]), generate=generate))
        generator = HFInsightGenerator({**self.profile, "max_input_tokens": 8, "max_new_tokens": 4})
        generator.runtime = runtime
        records = [row_input(row) for row in self.train.to_dict("records")[:4]]
        for record, text in zip(records, ("short", "long long long", "word " * 20, "medium medium")):
            record["essay"] = text
        results = generator.generate_batch(records)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["input_ids"].tolist(), [[1, 1, 2], [3, 3, 3], [1, 4, 4]])
        self.assertEqual(captured[0]["attention_mask"].tolist(), [[0, 0, 1], [1, 1, 1], [0, 1, 1]])
        self.assertFalse(captured[0]["do_sample"])
        self.assertIsInstance(results[2], ValueError)
        successful = [current_record for current_record in results if not isinstance(current_record, ValueError)]
        self.assertEqual([current_record[0] for current_record in successful], ["answer", "answer other", "answer other other other"])
        self.assertEqual([current_record[1]["input_tokens"] for current_record in successful], [1, 3, 2])
        self.assertEqual([current_record[1]["output_tokens"] for current_record in successful], [2, 3, 4])
        self.assertEqual([current_record[1]["stop_reason"] for current_record in successful], ["eos", "eos", "length"])
        self.assertTrue(all(current_record[1]["batch_size"] == 3 and current_record[1]["padded_input_tokens"] == 3 for current_record in successful))
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            generator(records[2])
        self.assertEqual(len(captured), 1)
        self.assertEqual(generator.generate_batch([]), [])

    def test_real_qwen_generation_matches_single_inputs_after_left_padding(self) -> None:
        """Verify that real qwen generation matches single inputs after left padding.

        Returns:
            None: This operation returns no value.
        """
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from offline_research.panda.InsightGenerator import PandaInsightGenerator

        torch.manual_seed(42)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=8, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096,
            bos_token_id=None, eos_token_id=1, pad_token_id=1)).eval()
        runtime: Any = PandaInsightGenerator("")
        runtime.model = model
        runtime.tokenizer = tiny_tokenizer()
        generator = HFInsightGenerator({**self.profile, "max_new_tokens": 8, "prompt_variant": "adjacent_text"})
        generator.runtime = runtime
        records = [row_input(row) for row in self.train.to_dict("records")[:3]]
        records[1]["essay"] += " word" * 30
        singles = [generator(record) for record in records]
        with patch.object(model, "generate", wraps=model.generate) as generate:
            batched = generator.generate_batch(records)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(generate.call_args.kwargs["input_ids"].shape[0], 3)
        successful = [current_record for current_record in batched if not isinstance(current_record, ValueError)]
        self.assertEqual(len(successful), len(records))
        self.assertEqual([current_record[0] for current_record in successful], [current_record[0] for current_record in singles])
        self.assertEqual([current_record[1]["input_tokens"] for current_record in successful], [current_record[1]["input_tokens"] for current_record in singles])
        self.assertEqual([current_record[1]["output_tokens"] for current_record in successful], [current_record[1]["output_tokens"] for current_record in singles])


if __name__ == "__main__":
    unittest.main()
