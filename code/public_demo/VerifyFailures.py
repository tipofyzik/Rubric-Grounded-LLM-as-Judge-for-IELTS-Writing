"""Controlled dependency failures over real HTTP; no production DB changes."""
from typing import Any

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from ielts_judge.Database import ReferenceRetriever, SupabaseRPC
from ielts_judge.ModelRegistry import ModelRegistry
from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from ielts_judge.Scoring import ScoringService


class EncoderDouble:
    """Manage encoder double through the operations defined by this component."""
    def __init__(self, profile: Any) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            profile (Any): Profile used by this operation.

        Returns:
            None: This operation returns no value.
        """
        self.profile = profile

    def encode(self, texts: Any) -> Any:
        """Encode.

        Args:
            texts (Any): Texts used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        return np.ones((len(texts), 768), dtype=np.float32) / np.sqrt(768), [20] * len(texts)


class JudgeDouble:
    """Manage judge double through the operations defined by this component."""
    def __init__(self) -> None:
        """Initialize this component's settings and collaborators.

        Returns:
            None: This operation returns no value.
        """
        self.registry = ModelRegistry()
        self.profile = self.registry.require_enabled("qwen25_1p5b")

    def count_input_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count the complete chat render with the selected judge tokenizer.

        Args:
            messages (list[dict[str, str]]): Ordered chat messages rendered with the selected tokenizer template.

        Returns:
            int: The computed result for the supplied inputs.
        """
        return 100

    def predict_messages(self, **keyword_arguments: Any) -> dict[str, Any]:
        """Generate a completion for validated chat messages within the shared deadline.

        Args:
            keyword_arguments (Any): Named arguments forwarded to the underlying implementation.

        Returns:
            dict[str, Any]: Named result fields and metadata produced by this operation.
        """
        return {"model_id": self.profile.repo_id, "model_revision": self.profile.revision,
                "prompt_version": "task2-scores-v2", "output_schema_version": "scores-integer-v2",
                "input_tokens": 100, "generated_tokens": 43, "max_new_tokens": 256, "stop_reason": "eos",
                "raw_text": json.dumps(dict.fromkeys(PUBLIC_CRITERION_IDS, 6))}


def main() -> None:
    """Parse command options and execute this component's command-line operation.

    Returns:
        None: This operation returns no value.
    """
    release = json.loads((ROOT / "results/stage4/release.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / release["corpus_path"] / "manifest.json").read_text(encoding="utf-8"))

    class Handler(BaseHTTPRequestHandler):
        """Manage handler through the operations defined by this component."""
        mode, calls = "transient", 0

        def log_message(self, *current_arguments: Any) -> None:
            """Log message.

            Args:
                current_arguments (Any): Current arguments used by this operation.

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
            Handler.calls += 1
            status, body = 503, {"code": "XX000", "message": "Controlled unavailable dependency"}
            if Handler.mode == "denied":
                status = 403
            elif Handler.mode == "cancelled":
                status, body = 500, {"code": "57014", "message": "canceling statement due to user request"}
            elif Handler.mode == "slow":
                time.sleep(0.3)
            elif Handler.mode == "ok":
                status, body = 200, {"corpus_id": manifest["corpus_id"],
                                    "embedding_profile_sha256": manifest["embedding_profile_sha256"],
                                    "count": manifest["count"], "references": []}
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                self.wfile.write(json.dumps(body).encode())
            except ConnectionError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    rpc = SupabaseRPC("http://127.0.0.1:" + str(server.server_port), "CONTROLLED_TEST_KEY", timeout=0.1)
    service = ScoringService(JudgeDouble(), retriever=ReferenceRetriever(rpc, manifest, EncoderDouble(manifest["embedding"])), setup="panda")
    results = []
    try:
        for mode, code, attempts in (("transient", "retrieval_failure", 2), ("denied", "retrieval_failure", 1),
                                     ("cancelled", "retrieval_cancelled", 1), ("slow", "retrieval_timeout", 1)):
            Handler.mode, Handler.calls = mode, 0
            result = service.score(model_key="qwen25_1p5b", task_prompt="Discuss public libraries.", essay="Libraries help people learn.")
            assert result["status"] == "error" and result["error"]["code"] == code
            assert Handler.calls == attempts and result["metadata"]["judge_attempts"] == 0
            assert "scores" not in result and result["metadata"]["retrieval"]["status"] == "error"
            results.append({"mode": mode, "observed_http_calls": Handler.calls, "result": result})
            Handler.mode, Handler.calls = "ok", 0
            recovered = service.score(model_key="qwen25_1p5b", task_prompt="Discuss public libraries.", essay="Libraries help people learn.")
            assert recovered["status"] == "ok" and recovered["warnings"] == ["retrieval_empty"]
            results[-1]["subsequent_request_recovered"] = True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    target = ROOT / "results/demo/verification/controlled-http-failures.json"
    target.write_text(json.dumps({"network": "real loopback HTTP", "model_doubles": True, "results": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"failure_scenarios": 4, "all_recovered": True, "production_db_changed": False}))


if __name__ == "__main__":
    main()
