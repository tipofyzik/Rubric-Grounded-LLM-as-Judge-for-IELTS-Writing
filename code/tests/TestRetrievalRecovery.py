"""Read-only HTTP recovery and safe service diagnostics; actual loopback HTTP."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest
import requests
from unittest.mock import Mock, patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "code/paths.yaml").is_file())
sys.path.insert(0, str(ROOT / "code"))
from ielts_judge.database import SupabaseRPC, RetrievalError


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.state = {"responses": [(200, {"ok": True})], "calls": 0}
        state = self.state
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
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

    def test_transient_read_failure_recovers_once(self):
        self.state["responses"] = [(500, {"code": "57014", "message": "canceling statement due to statement timeout"}),
                                    (200, {"ok": True})]
        self.assertEqual(self.rpc.call("panda_match", {}), {"ok": True})
        self.assertEqual(self.state["calls"], 2)

    def test_persistent_failure_is_bounded_and_safe(self):
        self.state["responses"] = [(500, {"code":"57014", "message":"canceling statement due to statement timeout",
                                          "details":"PRIVATE_ESSAY_OR_SECRET"})]
        with self.assertRaises(RetrievalError) as caught:
            self.rpc.call("panda_match", {})
        self.assertEqual(self.state["calls"], 2)
        self.assertEqual(caught.exception.code, "retrieval_timeout")
        self.assertEqual(caught.exception.diagnostics, {"http_status":500,"database_code":"57014","attempts":2})
        self.assertNotIn("PRIVATE", json.dumps(caught.exception.diagnostics))

    def test_authorization_and_client_errors_are_not_retried(self):
        for status in (301, 400, 401, 403, 429):
            self.state.update(calls=0, responses=[(status, {"code":"PGRST301"})])
            with self.subTest(status=status), self.assertRaises(RetrievalError) as caught:
                self.rpc.call("panda_match", {})
            self.assertEqual(self.state["calls"], 1)
            self.assertEqual(caught.exception.diagnostics["http_status"], status)

    def test_import_is_never_retried(self):
        self.state["responses"] = [(500, {"code":"XX000"}), (200, {"ok":True})]
        with self.assertRaises(RetrievalError):
            self.rpc.call("panda_import", {})
        self.assertEqual(self.state["calls"], 1)

    def test_explicit_database_cancellation_is_not_retried(self):
        self.state["responses"] = [(500, {"code":"57014", "message":"canceling statement due to user request"}),
                                    (200, {"ok":True})]
        with self.assertRaises(RetrievalError) as caught:
            self.rpc.call("panda_match", {})
        self.assertEqual(caught.exception.code, "retrieval_cancelled")
        self.assertEqual(self.state["calls"], 1)

    def test_transport_timeout_is_not_retried(self):
        with patch("ielts_judge.database.requests.post", side_effect=requests.Timeout("PRIVATE")) as post, \
             self.assertRaises(RetrievalError) as caught:
            self.rpc.call("panda_match", {})
        self.assertEqual(post.call_count, 1)
        self.assertEqual(caught.exception.code, "retrieval_timeout")
        self.assertEqual(caught.exception.diagnostics, {"attempts":1})
        self.assertNotIn("PRIVATE", str(caught.exception))

    def test_retry_keeps_the_original_time_budget(self):
        clock = [0.0]
        def first(*args, **kwargs):
            clock[0] = 3.0
            return Mock(status_code=500, text='{"code":"XX000"}')
        with patch("ielts_judge.database.time.monotonic", side_effect=lambda:clock[0]), \
             patch("ielts_judge.database.requests.post", side_effect=first) as post, \
             self.assertRaises(RetrievalError):
            self.rpc.call("panda_match", {}, timeout=1)
        self.assertEqual(post.call_count, 1)
        self.assertLessEqual(post.call_args.kwargs["timeout"], 1)

    def test_service_marks_attempted_retrieval_as_error(self):
        from TestScoringService import FakeAES, FakeJudge, FakeRetriever
        from ielts_judge.scoring import ScoringService
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
