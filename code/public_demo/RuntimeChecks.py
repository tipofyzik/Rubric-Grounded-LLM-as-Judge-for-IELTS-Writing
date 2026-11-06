"""Deployment boundaries using the real builder, runtime and installed imports."""
from typing import Any

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from SourceBuilder import prepare_source
from Runtime import HostedJudge, file_hash, validate_payload


class DeploymentTests(unittest.TestCase):
    """Manage deployment tests through the operations defined by this component."""
    def payload(self) -> dict[str, Any]:
        """Payload.

        Returns:
            dict[str, Any]: Named result fields and metadata produced by this operation.
        """
        return {"model_key": "qwen25_1p5b", "messages": [
            {"role": "system", "content": "Score the essay."},
            {"role": "user", "content": "Task and essay."}], "max_seconds": 15}

    def test_legacy_payload_still_works(self) -> None:
        """Verify that legacy payload still works.

        Returns:
            None: This operation returns no value.
        """
        now = time.monotonic()
        deadline = validate_payload(self.payload(), "qwen25_1p5b")
        self.assertGreater(deadline, now)
        self.assertLessEqual(deadline, now + 16)

    def test_waiting_does_not_restart_the_request_deadline(self) -> None:
        """Verify that waiting does not restart the request deadline.

        Returns:
            None: This operation returns no value.
        """
        payload = self.payload()
        payload["deadline"] = time.monotonic() - 1
        with self.assertRaises(TimeoutError):
            validate_payload(payload, "qwen25_1p5b")

    def test_external_payload_rejects_invalid_budgets_and_shapes(self) -> None:
        """Verify that external payload rejects invalid budgets and shapes.

        Returns:
            None: This operation returns no value.
        """
        for seconds in (True, 0, -1, 56, math.nan, math.inf, "15"):
            with self.subTest(seconds=seconds):
                payload = self.payload()
                payload["max_seconds"] = seconds
                with self.assertRaises(ValueError):
                    validate_payload(payload, "qwen25_1p5b")
        payload = self.payload()
        payload["file"] = "/etc/passwd"
        with self.assertRaises(ValueError):
            validate_payload(payload, "qwen25_1p5b")
        with self.assertRaises(ValueError):
            validate_payload(self.payload(), "another_model")

    def test_provider_titles_preserve_quota_and_timeout_results(self) -> None:
        """Verify that provider titles preserve quota and timeout results.

        Returns:
            None: This operation returns no value.
        """
        import gradio as gr
        from ielts_judge.Scoring import JudgeQuotaError

        for error, expected in (
            (gr.Error("Space app has reached its GPU limit.", title="ZeroGPU quota exceeded"), JudgeQuotaError),
            (gr.Error("No GPU was available", title="ZeroGPU queue timeout"), TimeoutError),
        ):
            with self.subTest(title=error.title):
                def reject(payload: Any) -> None:
                    """Reject.

                    Args:
                        payload (Any): Payload used by this operation.

                    Returns:
                        None: This operation returns no value.
                    """
                    raise error
                adapter = HostedJudge(SimpleNamespace(registry=None, profile=None, tokenizer=None), reject)
                with self.assertRaises(expected):
                    adapter.predict_messages(model_key="qwen25_1p5b", messages=[], deadline=time.monotonic() + 30)

    def test_public_bundle_contains_only_source_and_imports_in_isolation(self) -> None:
        """Verify that public bundle contains only source and imports in isolation.

        Returns:
            None: This operation returns no value.
        """
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            target = Path(directory)
            manifest = prepare_source(target, artifact_repo="owner/private-runtime",
                                      artifact_revision="a" * 40, artifact_manifest_sha256="b" * 64)
            names = {candidate_path.relative_to(target).as_posix() for candidate_path in target.rglob("*") if candidate_path.is_file()}
            self.assertEqual(names, set(manifest["files"]) | {"DEPLOYMENT.json", ".gitignore"})
            for name, expected in manifest["files"].items():
                self.assertEqual(file_hash(target / name), expected)
            self.assertFalse(any(current_count.startswith("results/") or current_count.endswith((".pt", ".env", ".jsonl")) for current_count in names))
            code = ('import sys;sys.path[:0] = ["code", "code/shared"];'
                    'from ielts_judge.ScoringRuntime import load_scoring_service;'
                    'from ielts_judge.Ui import build_demo;'
                    'from ielts_judge.llm.Judge import Judge;'
                    'print("isolated_imports_ok")')
            result = subprocess.run([sys.executable, "-B", "-I", "-c", code], cwd=target,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("isolated_imports_ok", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
