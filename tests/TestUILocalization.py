"""UI language behavior with real Gradio and an explicit scoring-service double."""
from typing import Any
import asyncio
import copy
from pathlib import Path
import string
import sys
from types import SimpleNamespace
import unittest

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from gradio.state_holder import SessionState
from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from ielts_judge.Localization import LANGUAGES, MESSAGES, translate
from ielts_judge.ModelRegistry import ModelRegistry
from ielts_judge.Ui import build_demo, load_examples, render_presentation, render_result


def success_result() -> dict[str, Any]:
    """Success result.

    Returns:
        dict[str, Any]: Named result fields and metadata produced by this operation.
    """
    return {"status": "ok", "run_id": "fixture-run", "scores": dict(zip(PUBLIC_CRITERION_IDS, (0, 6, 7, 9))),
            "metadata": {"aes": {"status": "ok", "input": {"truncated": True}},
                         "retrieval": {"status": "retrieval_empty", "used_count": 0, "query_truncated": True}},
            "timings": {"total_seconds": 1.25}, "warnings": ["retrieval_empty", "references_reduced_for_budget"]}


class FixtureService:
    """Only scoring is replaced. UI callbacks and session behavior are real."""
    def __init__(self) -> None:
        """Initialize this component's settings and collaborators.

        Returns:
            None: This operation returns no value.
        """
        registry = ModelRegistry()
        self.judge = SimpleNamespace(registry=registry, profile=registry.require_enabled("qwen25_1p5b"))
        self.limits = SimpleNamespace(max_input_chars=40000)
        self.calls = []
        self.gate = None

    def score(self, **keyword_arguments: Any) -> dict[str, Any]:
        """Evaluate one Task/Essay submission and return a bounded success or failure result.

        Args:
            keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

        Returns:
            dict[str, Any]: Status, four validated criterion scores or an error, and safe processing metadata.
        """
        self.calls.append(keyword_arguments)
        if self.gate is not None and not self.gate.wait(90):
            raise TimeoutError("Fixture gate expired")
        if not keyword_arguments["task_prompt"].strip() or not keyword_arguments["essay"].strip():
            return {"status": "error", "error": {"code": "invalid_input", "stage": "validation"}, "run_id": "fixture-error"}
        return success_result()


class UILocalizationTests(unittest.TestCase):
    """Manage uilocalization tests through the operations defined by this component."""
    def test_three_complete_catalogs_and_matching_format_fields(self) -> None:
        """Verify that three complete catalogs and matching format fields.

        Returns:
            None: This operation returns no value.
        """
        self.assertEqual([code for _, code in LANGUAGES], ["en", "ru", "de"])
        formatter = string.Formatter()
        for code, messages in MESSAGES.items():
            self.assertEqual(set(messages), set(MESSAGES["en"]), code)
            for key, value in messages.items():
                fields = {field for _, field, _, _ in formatter.parse(value) if field is not None}
                english = {field for _, field, _, _ in formatter.parse(MESSAGES["en"][key]) if field is not None}
                self.assertEqual(fields, english, (code, key))

    def test_english_input_hint_in_each_language(self) -> None:
        """Verify that english input hint in each language.

        Returns:
            None: This operation returns no value.
        """
        self.assertEqual(translate("en", "task"), "Task / Question")
        self.assertEqual(translate("en", "essay"), "Essay")
        for code, hint in [("ru", "(на английском)"), ("de", "(auf Englisch)")]:
            for key in ["task", "essay"]:
                self.assertIn(hint, translate(code, key))

    def test_results_and_warnings_translate_without_changing_data(self) -> None:
        """Verify that results and warnings translate without changing data.

        Returns:
            None: This operation returns no value.
        """
        result = success_result()
        original = copy.deepcopy(result)
        statuses = []
        for _, code in LANGUAGES:
            values = render_result(result, code)
            self.assertEqual(values[:4], ["0", "6", "7", "9"])
            self.assertIn(translate(code, "warning.retrieval_empty"), values[4])
            self.assertIn(translate(code, "warning.aes_truncated"), values[4])
            self.assertIn(translate(code, "warning.query_truncated"), values[4])
            self.assertIs(values[-1], result)
            statuses.append(values[5])
        self.assertEqual(len(set(statuses)), 3)
        self.assertEqual(result, original)

    def test_all_error_branches_have_localized_messages_and_no_scores(self) -> None:
        """Verify that all error branches have localized messages and no scores.

        Returns:
            None: This operation returns no value.
        """
        cases = [("invalid_input", "validation"), ("input_too_long", "preflight"),
                 ("context_budget_exceeded", "prompt"), ("service_busy", "admission"),
                 ("judge_quota", "judge"), ("invalid_output", "parser"),
                 ("judge_metadata_mismatch", "judge"), ("retrieval_timeout", "retrieval"),
                 ("retrieval_failure", "retrieval"), ("aes_failure", "aes"),
                 ("judge_failure", "judge"), ("unexpected_failure", "parser")]
        for code, stage in cases:
            result = {"status": "error", "error": {"code": code, "stage": stage}}
            messages = []
            for _, language in LANGUAGES:
                values = render_result(result, language)
                self.assertEqual(values[:4], [None] * 4)
                messages.append(values[5])
            self.assertEqual(len(set(messages)), 3, code)

    def test_examples_keep_their_english_content(self) -> None:
        """Verify that examples keep their english content.

        Returns:
            None: This operation returns no value.
        """
        examples = load_examples()
        self.assertEqual([example["id"] for example in examples], ["transport", "education", "remote_work"])
        for example in examples:
            labels = [translate(code, "example." + example["id"]) for _, code in LANGUAGES]
            self.assertEqual(len(set(labels)), 3)
            self.assertGreater(len(example["essay"].split()), 250)

    def test_unknown_language_falls_back_to_english(self) -> None:
        """Verify that unknown language falls back to english.

        Returns:
            None: This operation returns no value.
        """
        self.assertEqual(translate("unknown", "evaluate"), "Evaluate")

    def test_presentation_prepares_three_languages_without_changing_result(self) -> None:
        """Verify that presentation prepares three languages without changing result.

        Returns:
            None: This operation returns no value.
        """
        result = success_result()
        shown = render_presentation("ru", result)
        self.assertEqual(shown[:4], ["0", "6", "7", "9"])
        self.assertIs(shown[6], result)
        self.assertEqual(set(shown[7]), {"en", "ru", "de"})
        self.assertIn("Evaluation complete", shown[7]["en"][1])
        self.assertIn("Оценка завершена", shown[7]["ru"][1])
        self.assertIn("Bewertung abgeschlossen", shown[7]["de"][1])

    def test_real_gradio_sessions_keep_english_input_and_language_out_of_scoring(self) -> None:
        """Verify that real gradio sessions keep english input and language out of scoring.

        Returns:
            None: This operation returns no value.
        """
        service = FixtureService()
        async def check() -> None:
            """Check.

            Returns:
                None: This operation returns no value.
            """
            demo = build_demo(service)
            self.addCleanup(demo.close)
            ids = {current_field.api_name: index for index, current_field in demo.fns.items()}
            prepare_id, score_id = ids["prepare_submission"], ids["evaluate"]
            state_a, state_b = SessionState(demo), SessionState(demo)
            loaded = await demo.process_api(ids["load_example"], [0], state=state_a)
            for value in loaded["data"]:
                self.assertNotIn("label", value)
                self.assertNotIn("placeholder", value)
            self.assertEqual(loaded["data"][0]["value"], load_examples()[0]["task_prompt"])
            self.assertEqual(loaded["data"][1]["value"], load_examples()[0]["essay"])
            await demo.process_api(prepare_id, ["de", "qwen25_1p5b", "Original English task", "Original English essay"], state=state_a)
            await demo.process_api(prepare_id, ["ru", "qwen25_1p5b", "Other English task", "Other English essay"], state=state_b)
            state_id = demo.fns[score_id].inputs[0]._id
            self.assertEqual(state_a[state_id][0], "de")
            self.assertEqual(state_b[state_id][0], "ru")
            first = await demo.process_api(score_id, [None], state=state_a)
            final = await demo.process_api(score_id, [None], state=state_a, iterator=first["iterator"])
            self.assertEqual(service.calls, [{"model_key": "qwen25_1p5b", "task_prompt": "Original English task", "essay": "Original English essay"}])
            self.assertIn("Bewertung abgeschlossen", final["data"][5])
            self.assertEqual(set(final["data"][7].root), {"en", "ru", "de"})
            self.assertEqual(state_b[state_id][2], "Other English task")
            self.assertEqual(len(service.calls), 1)
        asyncio.run(check())


if __name__ == "__main__":
    unittest.main(verbosity=2)
