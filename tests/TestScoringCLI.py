"""Launcher and HTTP-stream handling checks; no real model/GPU calls."""
from typing import Any
from contextlib import redirect_stdout, redirect_stderr, ExitStack
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from local_backup import Score as launcher


class ManualLauncherTests(unittest.TestCase):
    """Manage manual launcher tests through the operations defined by this component."""
    def setUp(self) -> None:
        """SetUp.

        Returns:
            None: This operation returns no value.
        """
        self.folder = tempfile.TemporaryDirectory(dir=launcher.ROOT / ".cache")
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.task, self.essay = self.path / "task.txt", self.path / "essay.txt"
        self.task.write_text("My task", encoding="utf-8-sig")
        self.essay.write_text("PRIVATE_ESSAY\nSecond paragraph", encoding="utf-8-sig")
        self.service = Mock()
        self.service.score.return_value = {"status": "ok", "scores": {"task_response": 6}}

    def run_cli(self, current_arguments: Any) -> Any:
        """Run cli.

        Args:
            current_arguments (Any): Current arguments used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(launcher, "read_access", return_value={"HF_TOKEN": "NEVER_PRINT"}))
            stack.enter_context(patch.object(launcher, "preflight", return_value={"status":"ready","space":"https://example.invalid"}))
            judge = stack.enter_context(patch.object(launcher, "RemoteJudge"))
            judge.return_value.profile.key = "qwen25_1p5b"
            stack.enter_context(patch.object(launcher, "load_service", return_value=self.service))
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(redirect_stderr(io.StringIO()))
            code = launcher.main(current_arguments)
        return code, json.loads(output.getvalue()), judge

    def test_check_never_loads_models_or_scores(self) -> None:
        """Verify that check never loads models or scores.

        Returns:
            None: This operation returns no value.
        """
        code, result, judge = self.run_cli(["--check"])
        self.assertEqual((code, result["status"]), (0, "ready"))
        judge.assert_not_called()
        self.service.score.assert_not_called()

    def test_custom_files_preserve_paragraphs_and_output_excludes_input(self) -> None:
        """Verify that custom files preserve paragraphs and output excludes input.

        Returns:
            None: This operation returns no value.
        """
        destination = self.path / "result.json"
        code, result, _ = self.run_cli(["--task-file",str(self.task),"--essay-file",str(self.essay),"--output",str(destination)])
        self.assertEqual(code, 0)
        self.service.score.assert_called_once_with(model_key="qwen25_1p5b", task_prompt="My task", essay="PRIVATE_ESSAY\nSecond paragraph")
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), result)
        self.assertNotIn("PRIVATE_ESSAY", destination.read_text())
        self.assertNotIn("NEVER_PRINT", json.dumps(result))

    def test_existing_output_is_preserved_before_any_model_call(self) -> None:
        """Verify that existing output is preserved before any model call.

        Returns:
            None: This operation returns no value.
        """
        destination = self.path / "existing.json"
        destination.write_text("preserve", encoding="utf-8")
        code, result, judge = self.run_cli(["--output",str(destination)])
        self.assertEqual((code, result["error"]["code"]), (2, "output_exists"))
        self.assertEqual(destination.read_text(), "preserve")
        judge.assert_not_called()

    def test_single_input_file_is_rejected(self) -> None:
        """Verify that single input file is rejected.

        Returns:
            None: This operation returns no value.
        """
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            launcher.main(["--task-file",str(self.task)])
        self.assertEqual(error.exception.code, 2)

    def test_service_error_is_reported_with_nonzero_exit(self) -> None:
        """Verify that service error is reported with nonzero exit.

        Returns:
            None: This operation returns no value.
        """
        self.service.score.return_value = {"status":"error","error":{"code":"retrieval_failure"}}
        code, result, _ = self.run_cli([])
        self.assertEqual(code, 1)
        self.assertEqual(result["error"]["code"], "retrieval_failure")

    def stream(self, lines: Any) -> Any:
        """Stream.

        Args:
            lines (Any): Lines used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        judge = object.__new__(launcher.RemoteJudge)
        judge.base, judge.headers = "https://example.invalid/api", {"Authorization":"Bearer TEST_ONLY"}
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=None)
        submitted = Mock()
        submitted.json.return_value = {"event_id":"event"}
        session.post.return_value = submitted
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.iter_lines.return_value = lines
        session.get.return_value = response
        return judge, session

    def test_stream_completion_is_returned_without_buffering_prompts(self) -> None:
        """Verify that stream completion is returned without buffering prompts.

        Returns:
            None: This operation returns no value.
        """
        expected = {"raw_text":"{}","input_tokens":12}
        judge, session = self.stream(["event: heartbeat","data: null","event: complete","data: "+json.dumps([expected])])
        with patch.object(launcher.requests, "Session", return_value=session):
            actual = judge.predict_messages(model_key="qwen25_1p5b", messages=[], deadline=time.monotonic()+60)
        self.assertEqual(actual, expected)
        self.assertFalse(hasattr(judge, "messages"))
        self.assertFalse(hasattr(judge, "outputs"))

    def test_stream_quota_error_has_a_distinct_type(self) -> None:
        """Verify that stream quota error has a distinct type.

        Returns:
            None: This operation returns no value.
        """
        from ielts_judge.Scoring import JudgeQuotaError
        judge, session = self.stream(["event: error","data: GPU quota exceeded"])
        with patch.object(launcher.requests,"Session",return_value=session), self.assertRaises(JudgeQuotaError):
            judge.predict_messages(model_key="qwen25_1p5b", messages=[], deadline=time.monotonic()+60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
