"""Check local token handling and response persistence without downloading a model."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


project_root = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
temporary_folder = project_root / "code/offline_research/notebooks/.cache"
temporary_folder.mkdir(parents=True, exist_ok=True)
for variable in ("TEMP", "TMP", "TMPDIR"):
    os.environ[variable] = str(temporary_folder)
tempfile.tempdir = str(temporary_folder)
sys.dont_write_bytecode = True
os.environ["HF_HOME"] = str(temporary_folder / "TestCache/huggingface")
os.environ["TORCH_HOME"] = str(temporary_folder / "TestCache/torch")
os.environ["XDG_CACHE_HOME"] = str(temporary_folder / "TestCache")
sys.path[:0] = [str(project_root / "code"), str(project_root / "code/shared")]

import pandas as pd
import torch
from transformers.tokenization_utils_base import BatchEncoding

from offline_research.llm.LLMResponseObtainer import LLMResponseObtainer
from ielts_judge.llm.JsonEosBlocker import JsonEosBlocker
from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from ielts_judge.Paths import PATHS


class TestLLMResponseObtainer(unittest.TestCase):
    """Exercise the real response method with controlled generated token sequences."""

    @classmethod
    def setUpClass(cls) -> None:
        """Replace diagnostics in the fixed Kaggle test workspace.

        Returns:
            None: This operation returns no value.
        """
        cls.output_folder = (project_root / "code/offline_research/notebooks/artifacts/tests/responses").resolve()
        if not cls.output_folder.is_relative_to((project_root / "code/offline_research/notebooks/artifacts/tests").resolve()):
            raise ValueError("Test workspace escapes Kaggle artifacts")
        if cls.output_folder.exists():
            shutil.rmtree(cls.output_folder)
        cls.output_folder.mkdir(parents=True)

    def setUp(self) -> None:
        """Create one artifact folder and known expected scores for each test.

        Returns:
            None: This operation returns no value.
        """
        self.workspace = self.output_folder / self._testMethodName
        self.workspace.mkdir()
        self.submission = {"task_prompt": "Task", "essay": "Essay"}
        self.submissions = [{"essay_id": str(current_index), "task_prompt": "Task", "essay": f"Essay {current_index}"} for current_index in range(2)]
        self.expected = {"task_response": 6, "coherence_and_cohesion": 7,
                         "lexical_resource": 5, "grammatical_range_and_accuracy": 6}
        self.valid_text = json.dumps(self.expected, indent=2)
        log_file = self.enterContext((self.workspace / "Console.log").open("w", encoding="utf-8"))
        self.enterContext(redirect_stdout(log_file))

    def evaluator(self, response: str, tokens: list[int] | None = None,
                  limit: int = 384, eos: int | list[int] | None = 2) -> LLMResponseObtainer:
        """Attach a mocked tokenizer/model to the real evaluator.

        Args:
            response (str): Response used by this operation.
            tokens (list[int] | None): Tokens used by this operation.
            limit (int): Limit used by this operation.
            eos (int | list[int] | None): Eos used by this operation.

        Returns:
            LLMResponseObtainer: The computed result for the supplied inputs.
        """
        generated = [10, 11, 2] if tokens is None else tokens
        prompt_tokens = torch.tensor([[90, 91, 92]])
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<bos> formatted chat"
        tokenizer.return_value = BatchEncoding({
            "input_ids": prompt_tokens, "attention_mask": torch.ones_like(prompt_tokens),
        })
        tokenizer.decode.return_value = response
        tokenizer.eos_token_id = 2
        evaluator = LLMResponseObtainer(SERVICE_TOKEN="", max_new_tokens=limit)
        evaluator.tokenizer = tokenizer
        evaluator.model = SimpleNamespace(
            device=torch.device("cpu"), generation_config=SimpleNamespace(eos_token_id=eos),
            generate=Mock(return_value=torch.cat([prompt_tokens, torch.tensor([generated])], dim=1)),
        )
        return evaluator

    def test_only_generated_tokens_decoded(self) -> None:
        """Exclude prompt tokens and disable BPE cleanup and duplicate special tokens.

        Returns:
            None: This operation returns no value.
        """
        evaluator = self.evaluator(self.valid_text)
        result = evaluator._LLMResponseObtainer__evaluate_essay_locally(self.submission)
        self.assertEqual(result, self.expected)
        self.assertEqual(evaluator.tokenizer.decode.call_args.args[0].tolist(), [10, 11, 2])
        self.assertEqual(evaluator.tokenizer.decode.call_args.kwargs,
                         {"skip_special_tokens": True, "clean_up_tokenization_spaces": False})
        self.assertFalse(evaluator.tokenizer.call_args.kwargs["add_special_tokens"])
        self.assertEqual(evaluator.model.generate.call_args.kwargs["max_new_tokens"], 384)
        self.assertNotIn("temperature", evaluator.model.generate.call_args.kwargs)
        self.assertIs(evaluator.model.generate.call_args.kwargs["do_sample"], False)
        messages = evaluator.tokenizer.apply_chat_template.call_args.args[0]
        self.assertIn("### Essay task:\nTask", messages[1]["content"])

    def test_json_fence_is_accepted_but_role_prefix_is_rejected(self) -> None:
        """Verify that json fence is accepted but role prefix is rejected.

        Returns:
            None: This operation returns no value.
        """
        response = "```json\n" + self.valid_text + "\n```"
        evaluator = self.evaluator(response)
        self.assertEqual(evaluator._LLMResponseObtainer__evaluate_essay_locally(self.submission), self.expected)
        self.assertEqual(evaluator.last_generation["raw_text"], response)
        evaluator = self.evaluator("assistant\n" + response)
        self.assertIsNone(evaluator._LLMResponseObtainer__evaluate_essay_locally(self.submission))
        self.assertEqual(evaluator.last_generation["parse_error"], "invalid_integer_scores")

    def test_extra_string_field_with_braces_is_rejected(self) -> None:
        """Verify that extra string field with braces is rejected.

        Returns:
            None: This operation returns no value.
        """
        response = json.dumps({**self.expected, "note": "{a quoted value}"})
        evaluator = self.evaluator(response)
        self.assertIsNone(evaluator._LLMResponseObtainer__evaluate_essay_locally(self.submission))
        self.assertEqual(evaluator.last_generation["raw_text"], response)

    def test_limit_reached_without_repair(self) -> None:
        """Retain an incomplete JSON and identify exhaustion of the token budget.

        Returns:
            None: This operation returns no value.
        """
        response = "assistant\n" + self.valid_text.rstrip()[:-1]
        evaluator = self.evaluator(response, tokens=list(range(1000, 1384)))
        self.assertIsNone(evaluator._LLMResponseObtainer__evaluate_essay_locally(self.submission))
        self.assertEqual(evaluator.last_generation["stop_reason"], "max_new_tokens")
        self.assertEqual(evaluator.last_generation["generated_tokens"], 384)
        self.assertEqual(evaluator.last_generation["raw_text"], response)
        self.assertIsNotNone(evaluator.last_generation["parse_error"])

    def test_early_eos_without_repair(self) -> None:
        """Distinguish early EOS from a length limit for the same incomplete JSON.

        Returns:
            None: This operation returns no value.
        """
        evaluator = self.evaluator(self.valid_text.rstrip()[:-1], tokens=[10, 11, 12, 2])
        self.assertIsNone(evaluator._LLMResponseObtainer__evaluate_essay_locally(self.submission))
        self.assertEqual(evaluator.last_generation["stop_reason"], "eos")
        self.assertEqual(evaluator.last_generation["generated_tokens"], 4)

    def test_eos_at_limit(self) -> None:
        """Give explicit EOS priority when the final allowed token is EOS.

        Returns:
            None: This operation returns no value.
        """
        evaluator = self.evaluator(self.valid_text, tokens=[10, 11, 2], limit=3)
        self.assertEqual(evaluator._LLMResponseObtainer__evaluate_essay_locally(self.submission), self.expected)
        self.assertEqual(evaluator.last_generation["stop_reason"], "eos")

    def test_multiple_eos_tokens(self) -> None:
        """Recognize a model's end-of-turn token from its EOS list.

        Returns:
            None: This operation returns no value.
        """
        evaluator = self.evaluator(self.valid_text, tokens=[10, 11, 128009], eos=[128001, 128009])
        evaluator._LLMResponseObtainer__evaluate_essay_locally(self.submission)
        self.assertEqual(evaluator.last_generation["stop_reason"], "eos")

    def test_eos_blocking(self) -> None:
        """Block every EOS ID for incomplete JSON and preserve other token scores.

        Returns:
            None: This operation returns no value.
        """
        evaluator = self.evaluator(self.valid_text[:-1], eos=[2, 3])
        evaluator._LLMResponseObtainer__evaluate_essay_locally(self.submission)
        blocker = evaluator.model.generate.call_args.kwargs["logits_processor"][0]
        scores = torch.zeros((1, 12))
        result = blocker(torch.tensor([[90, 91, 92, 10]]), scores)
        self.assertEqual(blocker.prompt_length, 3)
        self.assertEqual(blocker.eos_tokens, [2, 3])
        self.assertTrue(torch.isneginf(result[0, [2, 3]]).all())
        self.assertTrue((result[0, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]] == 0).all())
        self.assertEqual(evaluator.tokenizer.decode.call_args.args[0].tolist(), [10])

    def test_complete_object(self) -> None:
        """Allow normal EOS after a complete object, including fenced responses.

        Returns:
            None: This operation returns no value.
        """
        for response in (self.valid_text, "assistant\n```json\n" + self.valid_text + "\n```",
                         json.dumps({**self.expected, "note": "brace } inside a string"})):
            with self.subTest(response=response):
                tokenizer = Mock()
                tokenizer.decode.return_value = response
                blocker = JsonEosBlocker(tokenizer, 1, [2, 3], PUBLIC_CRITERION_IDS)
                scores = torch.arange(12, dtype=torch.float).unsqueeze(0)
                original = scores.clone()
                result = blocker(torch.tensor([[90, 10]]), scores)
                self.assertTrue(torch.equal(result, original))

    def test_incomplete_object(self) -> None:
        """Reject balanced but incomplete objects and braces inside string values.

        Returns:
            None: This operation returns no value.
        """
        responses = ["", "No JSON", "{}", '{"task_achievement": 6}',
                     self.valid_text[:-1], '{"note": "}"}',
                     json.dumps({**self.expected, "lexical_resource": None}),
                     json.dumps({**self.expected, "lexical_resource": True}),
                     json.dumps({**self.expected, "lexical_resource": "5"})]
        for response in responses:
            with self.subTest(response=response):
                tokenizer = Mock()
                tokenizer.decode.return_value = response
                blocker = JsonEosBlocker(tokenizer, 1, [2], PUBLIC_CRITERION_IDS)
                result = blocker(torch.tensor([[90, 10]]), torch.zeros((1, 12)))
                self.assertTrue(torch.isneginf(result[0, 2]))

    def test_separate_sequences(self) -> None:
        """Block EOS independently for complete and incomplete batch responses.

        Returns:
            None: This operation returns no value.
        """
        tokenizer = Mock()
        tokenizer.decode.side_effect = [self.valid_text, self.valid_text[:-1]]
        blocker = JsonEosBlocker(tokenizer, 1, [2], PUBLIC_CRITERION_IDS)
        result = blocker(torch.tensor([[90, 10], [90, 11]]), torch.zeros((2, 12)))
        self.assertEqual(result[0, 2].item(), 0)
        self.assertTrue(torch.isneginf(result[1, 2]))

    def test_prompt_exclusion(self) -> None:
        """Keep EOS blocked when only the prompt contains a complete score object.

        Returns:
            None: This operation returns no value.
        """
        tokenizer = Mock()
        tokenizer.decode.return_value = ""
        blocker = JsonEosBlocker(tokenizer, 3, [2], PUBLIC_CRITERION_IDS)
        result = blocker(torch.tensor([[90, 91, 92]]), torch.zeros((1, 12)))
        self.assertTrue(torch.isneginf(result[0, 2]))
        self.assertEqual(tokenizer.decode.call_args.args[0].numel(), 0)

    def test_absent_eos(self) -> None:
        """Avoid changing generation when no EOS tokens are configured.

        Returns:
            None: This operation returns no value.
        """
        tokenizer = Mock()
        blocker = JsonEosBlocker(tokenizer, 1, [], PUBLIC_CRITERION_IDS)
        scores = torch.zeros((1, 12))
        self.assertIs(blocker(torch.tensor([[90, 10]]), scores), scores)
        tokenizer.decode.assert_not_called()

    def test_generation_continues(self) -> None:
        """Exercise real Transformers generation through early EOS to a JSON brace.

        Returns:
            None: This operation returns no value.
        """
        from transformers import GPT2Config, GPT2LMHeadModel

        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=16, n_positions=16, n_embd=8, n_layer=1, n_head=1,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )).eval()
        tokenizer = Mock()
        fragments = {3: self.valid_text.rstrip()[:-1], 4: "}", 2: ""}

        def decode_tokens(token_ids: torch.Tensor, **options: bool) -> str:
            """Decode controlled response fragments without downloading a tokenizer.

            Args:
                token_ids (torch.Tensor): Token ids used by this operation.
                options (bool): Options used by this operation.

            Returns:
                str: The computed result for the supplied inputs.
            """
            return "".join(fragments[token] for token in token_ids.tolist())

        def prefer_eos(input_ids: torch.LongTensor,
                       scores: torch.FloatTensor) -> torch.FloatTensor:
            """Make early EOS preferred over the closing brace after the JSON prefix.

            Args:
                input_ids (torch.LongTensor): Input ids used by this operation.
                scores (torch.FloatTensor): Four criterion scores using the public IELTS score contract.

            Returns:
                torch.FloatTensor: The computed result for the supplied inputs.
            """
            scores.fill_(-torch.inf)
            if input_ids.shape[1] == 1:
                scores[:, 3] = 0
            else:
                scores[:, 2] = 10
                scores[:, 4] = 9
            return scores

        tokenizer.decode.side_effect = decode_tokens
        prompt_tokens = torch.tensor([[1]])
        blocker = JsonEosBlocker(tokenizer, 1, [2], PUBLIC_CRITERION_IDS)
        for limit, expected in ((6, [1, 3, 4, 2]), (1, [1, 3])):
            with self.subTest(limit=limit), torch.no_grad():
                output = model.generate(
                    input_ids=prompt_tokens, attention_mask=torch.ones_like(prompt_tokens),
                    max_new_tokens=limit, do_sample=False,
                    logits_processor=[prefer_eos, blocker],
                )
                self.assertEqual(output[0].tolist(), expected)

    def test_prompt_object_not_used_as_answer(self) -> None:
        """Avoid extracting a valid object from a prompt when generation has none.

        Returns:
            None: This operation returns no value.
        """
        evaluator = self.evaluator("No answer")

        def decode_tokens(token_ids: torch.Tensor, **options: bool) -> str:
            """Return prompt JSON only when incorrectly asked to decode prompt IDs.

            Args:
                token_ids (torch.Tensor): Token ids used by this operation.
                options (bool): Options used by this operation.

            Returns:
                str: The computed result for the supplied inputs.
            """
            return self.valid_text if 90 in token_ids.tolist() else "No answer"

        evaluator.tokenizer.decode.side_effect = decode_tokens
        self.assertIsNone(evaluator._LLMResponseObtainer__evaluate_essay_locally(self.submission))
        self.assertEqual(evaluator.last_generation["parse_error"], "invalid_integer_scores")

    def test_loader_disables_cleanup(self) -> None:
        """Pass cleanup=False when loading and explicitly select evaluation mode.

        Returns:
            None: This operation returns no value.
        """
        model = Mock()
        evaluator = LLMResponseObtainer(SERVICE_TOKEN="")
        with patch("transformers.AutoTokenizer.from_pretrained") as tokenizer, \
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model):
            evaluator.load_model_and_tokenizer("test/model")
        tokenizer.assert_called_once_with("test/model", clean_up_tokenization_spaces=False)
        model.eval.assert_called_once()
        self.assertEqual(evaluator.max_new_tokens, 256)

    def test_per_sample_diagnostics(self) -> None:
        """Persist exact raw text and None for a bad answer alongside valid scores.

        Returns:
            None: This operation returns no value.
        """
        incomplete = self.valid_text.rstrip()[:-1]
        evaluator = self.evaluator(self.valid_text)
        evaluator.tokenizer.decode.side_effect = [self.valid_text, incomplete]
        with patch.dict(PATHS["results"], {"llm_hugging_face_responses": str(self.workspace)}):
            evaluator.get_response_locally(1, self.submissions)
        filename = PATHS["files"]["llm_response"].format(evaluation=1)
        parsed = json.loads((self.workspace / filename).read_text(encoding="utf-8"))
        diagnostic_path = (self.workspace / PATHS["files"]["llm_response_diagnostics"]
                           / (Path(filename).stem + ".jsonl"))
        records = [json.loads(line) for line in diagnostic_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["answer"] for row in parsed], [self.expected, None])
        self.assertEqual([row["essay_id"] for row in parsed], ["0", "1"])
        self.assertEqual([record["sample_id"] for record in records], ["0", "1"])
        self.assertEqual(records[1]["raw_text"], incomplete)
        self.assertIsNone(records[1]["prediction"])
        self.assertIsNotNone(records[1]["parse_error"])

    def test_diagnostics_survive_interruption(self) -> None:
        """Retain the completed sample's raw record if the next generation fails.

        Returns:
            None: This operation returns no value.
        """
        evaluator = self.evaluator(self.valid_text)
        generated = evaluator.model.generate.return_value
        evaluator.model.generate.side_effect = [generated, RuntimeError("Simulated interruption")]
        with patch.dict(PATHS["results"], {"llm_hugging_face_responses": str(self.workspace)}):
            with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                evaluator.get_response_locally(1, self.submissions)
        filename = PATHS["files"]["llm_response"].format(evaluation=1)
        diagnostic_path = (self.workspace / PATHS["files"]["llm_response_diagnostics"]
                           / (Path(filename).stem + ".jsonl"))
        records = diagnostic_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0])["prediction"], self.expected)


if __name__ == "__main__":
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestLLMResponseObtainer)
    )
    results_folder = getattr(TestLLMResponseObtainer, "output_folder", temporary_folder)
    (results_folder / "Results.txt").write_text(output.getvalue(), encoding="utf-8")
    summary = {"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "results": str(results_folder),
               "generation": "Controlled tokens and CPU Transformers generation with controlled logits; no pretrained model downloaded or GPU tested."}
    (results_folder / "Results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output.getvalue())
    print(json.dumps(summary, indent=2))
    sys.exit(not result.wasSuccessful())
