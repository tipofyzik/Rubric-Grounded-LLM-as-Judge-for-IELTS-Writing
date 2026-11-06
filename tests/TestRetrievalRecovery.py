"""Read-only HTTP recovery and safe service diagnostics; actual loopback HTTP."""
from typing import Any
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest
import requests
from unittest.mock import Mock, patch

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from ielts_judge.Database import SupabaseRPC, RetrievalError


class RecoveryTests(unittest.TestCase):
    """Manage recovery tests through the operations defined by this component."""
    def setUp(self) -> None:
        """SetUp.

        Returns:
            None: This operation returns no value.
        """
        self.state = {"responses": [(200, {"ok": True})], "calls": 0}
        state = self.state
        class Handler(BaseHTTPRequestHandler):
            """Manage handler through the operations defined by this component."""
            def log_message(self, *_: Any) -> None:
                """Log message.

                Args:
                    _ (Any):   used by this operation.

                Returns:
                    None: This operation returns no value.
                """
                pass

            def do_POST(self) -> None:
                """Do POST.

                Returns:
                    None: This operation returns no value.
                """
                self.rfile.read(int(self.headers["Content-Length"]))
                index = min(state["calls"], len(state["responses"]) - 1)
                status, body = state["responses"][index]
                state["calls"] += 1
                self.send_response(status)
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.rpc = SupabaseRPC(f"http://127.0.0.1:{self.server.server_port}", "sb_publishable_test", timeout=2)

    def test_transient_read_failure_recovers_once(self) -> None:
        """Verify that transient read failure recovers once.

        Returns:
            None: This operation returns no value.
        """
        self.state["responses"] = [(500, {"code": "57014", "message": "canceling statement due to statement timeout"}),
                                    (200, {"ok": True})]
        self.assertEqual(self.rpc.call("panda_match", {}), {"ok": True})
        self.assertEqual(self.state["calls"], 2)

    def test_persistent_failure_is_bounded_and_safe(self) -> None:
        """Verify that persistent failure is bounded and safe.

        Returns:
            None: This operation returns no value.
        """
        self.state["responses"] = [(500, {"code":"57014", "message":"canceling statement due to statement timeout",
                                          "details":"PRIVATE_ESSAY_OR_SECRET"})]
        with self.assertRaises(RetrievalError) as caught:
            self.rpc.call("panda_match", {})
        self.assertEqual(self.state["calls"], 2)
        self.assertEqual(caught.exception.code, "retrieval_timeout")
        self.assertEqual(caught.exception.diagnostics, {"http_status":500,"database_code":"57014","attempts":2})
        self.assertNotIn("PRIVATE", json.dumps(caught.exception.diagnostics))

    def test_authorization_and_client_errors_are_not_retried(self) -> None:
        """Verify that authorization and client errors are not retried.

        Returns:
            None: This operation returns no value.
        """
        for status in (301, 400, 401, 403, 429):
            self.state.update(calls=0, responses=[(status, {"code":"PGRST301"})])
            with self.subTest(status=status), self.assertRaises(RetrievalError) as caught:
                self.rpc.call("panda_match", {})
            self.assertEqual(self.state["calls"], 1)
            self.assertEqual(caught.exception.diagnostics["http_status"], status)

    def test_import_is_never_retried(self) -> None:
        """Verify that import is never retried.

        Returns:
            None: This operation returns no value.
        """
        self.state["responses"] = [(500, {"code":"XX000"}), (200, {"ok":True})]
        with self.assertRaises(RetrievalError):
            self.rpc.call("panda_import", {})
        self.assertEqual(self.state["calls"], 1)

    def test_explicit_database_cancellation_is_not_retried(self) -> None:
        """Verify that explicit database cancellation is not retried.

        Returns:
            None: This operation returns no value.
        """
        self.state["responses"] = [(500, {"code":"57014", "message":"canceling statement due to user request"}),
                                    (200, {"ok":True})]
        with self.assertRaises(RetrievalError) as caught:
            self.rpc.call("panda_match", {})
        self.assertEqual(caught.exception.code, "retrieval_cancelled")
        self.assertEqual(self.state["calls"], 1)

    def test_transport_timeout_is_not_retried(self) -> None:
        """Verify that transport timeout is not retried.

        Returns:
            None: This operation returns no value.
        """
        with patch("ielts_judge.Database.requests.post", side_effect=requests.Timeout("PRIVATE")) as post, \
             self.assertRaises(RetrievalError) as caught:
            self.rpc.call("panda_match", {})
        self.assertEqual(post.call_count, 1)
        self.assertEqual(caught.exception.code, "retrieval_timeout")
        self.assertEqual(caught.exception.diagnostics, {"attempts":1})
        self.assertNotIn("PRIVATE", str(caught.exception))

    def test_retry_keeps_the_original_time_budget(self) -> None:
        """Verify that retry keeps the original time budget.

        Returns:
            None: This operation returns no value.
        """
        clock = [0.0]
        def first(*current_arguments: Any, **keyword_arguments: Any) -> Any:
            """First.

            Args:
                current_arguments (Any): Current arguments used by this operation.
                keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

            Returns:
                Any: The computed result for the supplied inputs.
            """
            clock[0] = 3.0
            return Mock(status_code=500, text='{"code":"XX000"}')
        with patch("ielts_judge.Database.time.monotonic", side_effect=lambda:clock[0]), \
             patch("ielts_judge.Database.requests.post", side_effect=first) as post, \
             self.assertRaises(RetrievalError):
            self.rpc.call("panda_match", {}, timeout=1)
        self.assertEqual(post.call_count, 1)
        self.assertLessEqual(post.call_args.kwargs["timeout"], 1)

    def test_service_marks_attempted_retrieval_as_error(self) -> None:
        """Verify that service marks attempted retrieval as error.

        Returns:
            None: This operation returns no value.
        """
        from TestScoringService import FakeAES, FakeJudge, FakeRetriever
        from ielts_judge.Scoring import ScoringService
        retriever, judge = FakeRetriever(), FakeJudge()
        retriever.error = RetrievalError("PRIVATE", diagnostics={"http_status":500,"database_code":"57014","attempts":2})
        service = ScoringService(judge, aes=FakeAES(), retriever=retriever)
        result = service.score(model_key="qwen25_1p5b", task_prompt="Task", essay="Essay")
        self.assertEqual(result["metadata"]["retrieval"]["status"], "error")
        self.assertEqual(result["error"]["http_status"], 500)
        self.assertEqual(result["error"]["database_code"], "57014")
        self.assertEqual(result["metadata"]["judge_attempts"], 0)
        self.assertNotIn("PRIVATE", json.dumps(result))
        retriever.error = None
        self.assertEqual(service.score(model_key="qwen25_1p5b",task_prompt="Task",essay="Essay")["status"], "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
