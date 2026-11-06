"""Scoring contracts: real render/parser/LangChain; model doubles are explicit."""
from typing import Any
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from ielts_judge.Database import EncodingError, RetrievalError
from ielts_judge.ModelRegistry import ModelRegistry
from ielts_judge.Prompts import CONTEXT_SETUPS, INSTRUCTION, OUTPUT, JudgeContext, ReferenceContext, build_judge_messages
from ielts_judge.Scores import OUTPUT_SCHEMA_VERSION, parse_scores
from ielts_judge.Scoring import ScoringService, ScoringLimits, JudgeQuotaError

SCORES = dict(zip(PUBLIC_CRITERION_IDS, (0, 6, 7, 9)))
REFERENCE = ReferenceContext("HIDDEN_ID", "HIDDEN_TASK", SCORES,
                             "Original failed/partial insight.\nEvidence is preserved.", "HIDDEN_ESSAY")


class FakeJudge:
    """Only the weights/tokenizer are replaced; all service/prompt/parser logic runs."""
    def __init__(self) -> None:
        """Initialize this component's settings and collaborators.

        Returns:
            None: This operation returns no value.
        """
        self.registry = ModelRegistry()
        self.profile = self.registry.require_enabled("qwen25_1p5b")
        self.raw = json.dumps(SCORES)
        self.calls = []
        self.error = None
        self.token_count = lambda messages: len(messages[1]["content"]) // 4

    def count_input_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count the complete chat render with the selected judge tokenizer.

        Args:
            messages (list[dict[str, str]]): Ordered chat messages rendered with the selected tokenizer template.

        Returns:
            int: The computed result for the supplied inputs.
        """
        return self.token_count(messages)

    def predict_messages(self, *, model_key: str, messages: list[dict[str, str]], deadline: float) -> dict[str, Any]:
        """Generate a completion for validated chat messages within the shared deadline.

        Args:
            model_key (str): Enabled immutable judge profile key from the model registry.
            messages (list[dict[str, str]]): Ordered chat messages rendered with the selected tokenizer template.
            deadline (float): Absolute monotonic request deadline shared by the processing stages.

        Returns:
            dict[str, Any]: Named result fields and metadata produced by this operation.
        """
        from langsmith.utils import tracing_is_enabled
        self.calls.append(messages)
        if tracing_is_enabled():
            raise AssertionError("User request tracing is enabled")
        if self.error:
            raise self.error
        return {"raw_text": self.raw, "model_id": self.profile.repo_id,
                "model_revision": self.profile.revision, "prompt_version": "task2-scores-v2",
                "output_schema_version": OUTPUT_SCHEMA_VERSION,
                "input_tokens": self.count_input_tokens(messages), "generated_tokens": 45,
                "max_new_tokens": 256, "stop_reason": "eos"}


class FakeAES:
    """Manage fake aes through the operations defined by this component."""
    def __init__(self) -> None:
        """Initialize this component's settings and collaborators.

        Returns:
            None: This operation returns no value.
        """
        self.calls = []
        self.error = None

    def predict_essay(self, essay: str) -> dict[str, Any]:
        """Return essay-only AES estimates with the model's input-length metadata.

        Args:
            essay (str): English essay text supplied for evaluation.

        Returns:
            dict[str, Any]: Named result fields and metadata produced by this operation.
        """
        self.calls.append(essay)
        if self.error:
            raise self.error
        return {"scores": SCORES, "input": {"original_tokens": 600, "used_tokens": 512,
                "max_length": 512, "truncated": True}, "semantics": "essay_only; not task-aware TR"}


class FakeRetriever:
    """Manage fake retriever through the operations defined by this component."""
    def __init__(self) -> None:
        """Initialize this component's settings and collaborators.

        Returns:
            None: This operation returns no value.
        """
        self.calls = []
        self.error = None
        self.references = (REFERENCE,)

    def retrieve(self, essay: str, **keyword_arguments: Any) -> dict[str, Any]:
        """Encode the query essay and retrieve compatible training references with ID/hash exclusion.

        Args:
            essay (str): English essay text supplied for evaluation.
            keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

        Returns:
            dict[str, Any]: Named result fields and metadata produced by this operation.
        """
        self.calls.append((essay, keyword_arguments))
        if self.error:
            raise self.error
        return {"status": "ok" if self.references else "retrieval_empty", "references": self.references,
                "corpus_id": "c" * 64, "similarities": [0.8] * len(self.references),
                "query_original_tokens": 600, "query_used_tokens": 512, "query_truncated": True,
                "encoding_seconds": 0.01, "rpc_seconds": 0.02}


class ParserTests(unittest.TestCase):
    """Manage parser tests through the operations defined by this component."""
    def test_allowed_formats_preserve_values(self) -> None:
        """Verify that allowed formats preserve values.

        Returns:
            None: This operation returns no value.
        """
        floats = {current_key: float(current_value) for current_key, current_value in SCORES.items()}
        for text in (json.dumps(SCORES), " \n" + json.dumps(floats) + "\n ",
                     "```json\n" + json.dumps(floats) + "\n```", "```\n" + json.dumps(SCORES) + "\n```"):
            self.assertEqual(parse_scores(text), SCORES)
            self.assertTrue(all(type(current_value) is int for current_value in parse_scores(text).values()))

    def test_invalid_values_fields_and_ambiguous_text(self) -> None:
        """Verify that invalid values fields and ambiguous text.

        Returns:
            None: This operation returns no value.
        """
        bad = [json.dumps({**SCORES, "task_response": sample})
               for sample in (True, None, "6", 6.5, -1, 10, float("nan"), float("inf"), 10**400)]
        bad += [json.dumps({**SCORES, "extra": 6}), json.dumps({current_key:current_value for current_key,current_value in SCORES.items() if current_key != "task_response"}),
                json.dumps(SCORES).replace('"task_response": 0', '"task_response": 6.0000000000000001'),
                json.dumps(SCORES).replace('"task_response": 0', '"task_response": 1e999999999999999999999999999999'),
                "[]", "Here are the scores: " + json.dumps(SCORES), json.dumps(SCORES) + " trailing",
                json.dumps(SCORES) + json.dumps(SCORES), '{"task_response":0,"task_response":6}',
                '```json\n' + json.dumps(SCORES) + '\n```\nmore', '{"task_response": 6']
        for text in bad:
            with self.subTest(text=text[:80]), self.assertRaises(ValueError):
                parse_scores(text)


class PromptTests(unittest.TestCase):
    """Manage prompt tests through the operations defined by this component."""
    def test_research_setups_use_the_same_prompt_without_dataset_targets(self) -> None:
        """Verify that research setups use the same prompt without dataset targets.

        Returns:
            None: This operation returns no value.
        """
        from offline_research.llm.LLMResponseObtainer import LLMResponseObtainer
        submission = {"task_prompt":"Task", "essay":"Essay", "forbidden_gold":"DO_NOT_LEAK"}
        for setup, (rubric, current_references, aes) in CONTEXT_SETUPS.items():
            supplied = JudgeContext(SCORES, (REFERENCE,))
            obtainer = LLMResponseObtainer("", setup=setup, context_provider=lambda _: supplied)
            messages = obtainer._messages(submission)
            expected = build_judge_messages("Task", "Essay", JudgeContext(
                SCORES if aes else {}, (REFERENCE,) if current_references else (), rubric))
            self.assertEqual(messages, expected)
            self.assertNotIn("DO_NOT_LEAK", json.dumps(messages))

    def test_fine_tuned_evaluation_executes_common_prompt_and_parser(self) -> None:
        """Verify that fine tuned evaluation executes common prompt and parser.

        Returns:
            None: This operation returns no value.
        """
        from offline_research.llm.LLMFineTuner import LLMFineTuner
        from ielts_judge.llm.Generation import GenerationResult
        tuner = object.__new__(LLMFineTuner)
        tuner.model = SimpleNamespace(eval=lambda:None)
        tuner.tokenizer = object()
        submission = {"task_prompt":"Task", "essay":"Essay", "essay_id":"test-1", "dataset_manifest":"m"}
        generated = GenerationResult("```json\n" + json.dumps({current_key:float(current_value) for current_key,current_value in SCORES.items()}) + "\n```",
                                     100, 45, 256, 2, "eos")
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory, \
             patch("offline_research.llm.LLMFineTuner.generate_response", return_value=generated) as generate:
            result = tuner.evaluate_llm([submission], [[9]*4], output_file=str(Path(directory) / "fine-tuned-check.json"))
        self.assertEqual(result[0]["prediction_json"], SCORES)
        self.assertEqual(generate.call_args.args[2], build_judge_messages("Task","Essay",JudgeContext(include_rubric=False)))
        self.assertNotIn('"true_scores"', json.dumps(generate.call_args.args[2]))

    def test_six_renders_keep_instruction_output_and_text(self) -> None:
        """Verify that six renders keep instruction output and text.

        Returns:
            None: This operation returns no value.
        """
        for name, (rubric, current_references, aes) in CONTEXT_SETUPS.items():
            context = JudgeContext(SCORES if aes else {}, (REFERENCE,) if current_references else (), rubric)
            messages = build_judge_messages("CURRENT_TASK", "CURRENT_ESSAY\nKEEP_PARAGRAPH", context)
            self.assertEqual(messages[0]["content"], INSTRUCTION)
            text = messages[1]["content"]
            self.assertTrue(text.endswith(OUTPUT))
            self.assertIn("CURRENT_ESSAY\nKEEP_PARAGRAPH", text)
            for heading, enabled in (("### Rubric:", rubric), ("### Reference insights:", current_references), ("### Preliminary scores:", aes)):
                self.assertEqual(heading in text, enabled, name)
            for hidden in ("HIDDEN_ID", "HIDDEN_TASK", "HIDDEN_ESSAY", "diagnostic", "step 0.5"):
                self.assertNotIn(hidden, text)
            if current_references:
                self.assertIn(REFERENCE.insights, text)

    def test_multiple_examples_are_numbered_and_blocks_ordered(self) -> None:
        """Verify that multiple examples are numbered and blocks ordered.

        Returns:
            None: This operation returns no value.
        """
        text = build_judge_messages("Task", "Essay", JudgeContext(SCORES, (REFERENCE, REFERENCE)))[1]["content"]
        self.assertIn("Example 1", text)
        self.assertIn("Example 2", text)
        positions = [text.index(current_digest) for current_digest in ("### Essay task:", "### Essay:", "### Rubric:", "### Reference insights:", "### Preliminary scores:", "### Output:")]
        self.assertEqual(positions, sorted(positions))


class ServiceTests(unittest.TestCase):
    """Manage service tests through the operations defined by this component."""
    def setUp(self) -> None:
        """SetUp.

        Returns:
            None: This operation returns no value.
        """
        self.judge, self.aes, self.retriever = FakeJudge(), FakeAES(), FakeRetriever()
        self.service = ScoringService(self.judge, aes=self.aes, retriever=self.retriever)

    def score(self, **keyword_arguments: Any) -> dict[str, Any]:
        """Evaluate one Task/Essay submission and return a bounded success or failure result.

        Args:
            keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

        Returns:
            dict[str, Any]: Status, four validated criterion scores or an error, and safe processing metadata.
        """
        return self.service.score(**({"model_key": "qwen25_1p5b", "task_prompt": "Task", "essay": "PRIVATE_ESSAY"} | keyword_arguments))

    def test_full_chain_returns_scores_and_metadata_without_content(self) -> None:
        """Verify that full chain returns scores and metadata without content.

        Returns:
            None: This operation returns no value.
        """
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}):
            result = self.score()
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["scores"], SCORES)
        self.assertEqual(self.aes.calls, ["PRIVATE_ESSAY"])
        self.assertEqual(self.retriever.calls[0][0], "PRIVATE_ESSAY")
        self.assertEqual(len(self.judge.calls), 1)
        self.assertTrue(result["metadata"]["aes"]["input"]["truncated"])
        self.assertEqual(result["metadata"]["retrieval"]["used_reference_ids"], ["HIDDEN_ID"])
        self.assertNotIn("PRIVATE_ESSAY", json.dumps(result))
        self.assertNotIn("Original failed", json.dumps(result))

    def test_all_setups_call_only_enabled_components(self) -> None:
        """Verify that all setups call only enabled components.

        Returns:
            None: This operation returns no value.
        """
        for setup, (_, current_references, aes) in CONTEXT_SETUPS.items():
            self.service = ScoringService(self.judge, aes=self.aes, retriever=self.retriever, setup=setup)
            self.aes.calls.clear(); self.retriever.calls.clear()
            self.assertEqual(self.score()["status"], "ok")
            self.assertEqual(bool(self.aes.calls), aes)
            self.assertEqual(bool(self.retriever.calls), current_references)

    def test_invalid_input_never_runs_models(self) -> None:
        """Verify that invalid input never runs models.

        Returns:
            None: This operation returns no value.
        """
        for keyword_arguments in ({"task_prompt":" "}, {"essay":None}, {"model_key":"unknown"}, {"model_key":[]}):
            result = self.score(**keyword_arguments)
            self.assertEqual(result["error"]["code"], "invalid_input")
        self.assertFalse(self.judge.calls or self.aes.calls or self.retriever.calls)

    def test_preflight_rejects_overflow_before_aes(self) -> None:
        """Verify that preflight rejects overflow before aes.

        Returns:
            None: This operation returns no value.
        """
        self.judge.token_count = lambda _: 3841
        self.assertEqual(self.score()["error"]["code"], "input_too_long")
        self.assertFalse(self.aes.calls)

    def test_budget_drops_whole_tail_reference_and_never_essay(self) -> None:
        """Verify that budget drops whole tail reference and never essay.

        Returns:
            None: This operation returns no value.
        """
        self.retriever.references = (REFERENCE, ReferenceContext("tail", "t", SCORES, "DROP_THIS"))
        self.judge.token_count = lambda current_match: 3841 if "DROP_THIS" in current_match[1]["content"] else 3840
        result = self.score()
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["metadata"]["retrieval"]["dropped_reference_ids"], ["tail"])
        self.assertIn("PRIVATE_ESSAY", self.judge.calls[0][1]["content"])
        self.assertIn(REFERENCE.insights, self.judge.calls[0][1]["content"])

    def test_no_reference_fits_is_explicit_and_does_not_generate(self) -> None:
        """Verify that no reference fits is explicit and does not generate.

        Returns:
            None: This operation returns no value.
        """
        self.judge.token_count = lambda current_match: 3841 if "### Reference insights:" in current_match[1]["content"] else 100
        result = self.score()
        self.assertEqual(result["error"]["code"], "context_budget_exceeded")
        self.assertFalse(self.judge.calls)
        self.assertEqual(result["metadata"]["judge_attempts"], 0)

    def test_empty_retrieval_is_visible_and_continues(self) -> None:
        """Verify that empty retrieval is visible and continues.

        Returns:
            None: This operation returns no value.
        """
        self.retriever.references = ()
        result = self.score()
        self.assertEqual(result["status"], "ok")
        self.assertIn("retrieval_empty", result["warnings"])
        self.assertNotIn("### Reference insights:", self.judge.calls[0][1]["content"])

    def test_dependency_failures_and_next_request_recovery(self) -> None:
        """Verify that dependency failures and next request recovery.

        Returns:
            None: This operation returns no value.
        """
        for component, error, code in (
            (self.aes, RuntimeError("private detail"), "aes_failure"),
            (self.retriever, EncodingError("private detail"), "encoder_failure"),
            (self.retriever, RetrievalError("private detail"), "retrieval_failure"),
            (self.retriever, RetrievalError("private detail", "retrieval_timeout"), "retrieval_timeout"),
            (self.judge, JudgeQuotaError("private detail"), "judge_quota"),
            (self.judge, TimeoutError("private detail"), "judge_timeout"),
            (self.judge, RuntimeError("private detail"), "judge_failure")):
            component.error = error
            result = self.score()
            self.assertEqual(result["error"]["code"], code, result)
            self.assertNotIn("private detail", json.dumps(result))
            self.assertNotIn("scores", result)
            component.error = None
            self.assertEqual(self.score()["status"], "ok")

    def test_invalid_output_does_not_retry_or_impute(self) -> None:
        """Verify that invalid output does not retry or impute.

        Returns:
            None: This operation returns no value.
        """
        self.judge.raw = '{"task_response": 6}'
        result = self.score()
        self.assertEqual(result["error"]["code"], "invalid_output")
        self.assertEqual(len(self.judge.calls), 1)
        self.assertNotIn("scores", result)

    def test_expired_stage_prevents_later_work(self) -> None:
        """Verify that expired stage prevents later work.

        Returns:
            None: This operation returns no value.
        """
        clock = [0.0]
        original = self.aes.predict_essay
        def expire(essay: str) -> Any:
            """Expire.

            Args:
                essay (str): English essay text supplied for evaluation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            value = original(essay)
            clock[0] = 121.0
            return value
        self.aes.predict_essay = expire
        with patch("ielts_judge.Scoring.time.monotonic", side_effect=lambda: clock[0]):
            result = self.score()
        self.assertEqual(result["error"]["code"], "aes_timeout")
        self.assertFalse(self.retriever.calls or self.judge.calls)
        self.aes.predict_essay = original
        self.assertEqual(self.score()["status"], "ok")

    def test_concurrent_request_is_rejected_without_shared_request_state(self) -> None:
        """Verify that concurrent request is rejected without shared request state.

        Returns:
            None: This operation returns no value.
        """
        entered, release = threading.Event(), threading.Event()
        original = self.aes.predict_essay
        def wait(essay: str) -> Any:
            """Wait.

            Args:
                essay (str): English essay text supplied for evaluation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            entered.set()
            if not release.wait(5):
                raise AssertionError("Test synchronization failed")
            return original(essay)
        self.aes.predict_essay = wait
        outputs = []
        thread = threading.Thread(target=lambda: outputs.append(self.score(essay="FIRST")))
        thread.start()
        self.assertTrue(entered.wait(5))
        try:
            second = self.score(essay="SECOND")
            self.assertEqual(second["error"]["code"], "service_busy")
        finally:
            release.set(); thread.join(5)
        self.assertEqual(outputs[0]["status"], "ok")
        self.assertNotIn("SECOND", self.judge.calls[0][1]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
