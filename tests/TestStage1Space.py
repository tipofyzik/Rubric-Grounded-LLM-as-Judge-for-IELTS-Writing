"""Check the manual probe and its exact deployable core without hosted access."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from TestJudge import JudgeTestCase, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "code/offline_research/legacy/stage1_space"))

from ielts_judge.llm.Judge import Judge
from Probe import run_probe
from offline_research.tools.BuildStage1Space import APP_FILES, CORE_FILES, build_bundle


class TestStage1Space(JudgeTestCase):
    """Manage test stage1space through the operations defined by this component."""
    def test_manual_input_uses_real_judge_and_each_click_generates(self) -> None:
        """Verify that manual input uses real judge and each click generates.

        Returns:
            None: This operation returns no value.
        """
        model, tokenizer = self.runtime()
        judge = Judge(self.profile.key, model, tokenizer, self.registry)
        first = run_probe(judge, model_key=self.profile.key, task_prompt="Task A", essay="Essay")
        repeat = run_probe(judge, model_key=self.profile.key, task_prompt="Task A", essay="Essay")
        changed = run_probe(judge, model_key=self.profile.key, task_prompt="Task B", essay="Essay")
        changed_essay = run_probe(judge, model_key=self.profile.key, task_prompt="Task B", essay="Other essay")
        self.assertEqual(first["raw_text"], self.response)
        self.assertEqual(first["submission_sha256"], repeat["submission_sha256"])
        self.assertNotEqual(first["probe_run_id"], repeat["probe_run_id"])
        self.assertNotEqual(first["submission_sha256"], changed["submission_sha256"])
        self.assertNotEqual(changed["submission_sha256"], changed_essay["submission_sha256"])
        self.assertEqual(first["context_mode"], "empty")
        self.assertEqual(first["reference_ids"], [])
        self.assertEqual(model.generate.call_count, 4)

    def test_synthetic_context_reaches_prompt_and_is_identified(self) -> None:
        """Verify that synthetic context reaches prompt and is identified.

        Returns:
            None: This operation returns no value.
        """
        model, tokenizer = self.runtime()
        judge = Judge(self.profile.key, model, tokenizer, self.registry)
        result = run_probe(judge, model_key=self.profile.key, task_prompt="Task", essay="Essay",
                           use_synthetic_context=True)
        payload = tokenizer.apply_chat_template.call_args.args[0][1]["content"]
        self.assertEqual(result["context_mode"], "synthetic_fixture")
        self.assertEqual(result["reference_ids"], ["synthetic-reference-1"])
        self.assertFalse(result["uses_real_aes_or_retrieval"])
        self.assertIn("Example 1", payload)
        self.assertNotIn("synthetic-reference-1", payload)
        self.assertIn("essay-only AES estimates", payload)
        self.assertIn("task_response", payload)

    def test_invalid_manual_request_does_not_generate(self) -> None:
        """Verify that invalid manual request does not generate.

        Returns:
            None: This operation returns no value.
        """
        model, tokenizer = self.runtime()
        judge = Judge(self.profile.key, model, tokenizer, self.registry)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            run_probe(judge, model_key=self.profile.key, task_prompt="", essay="Essay")
        model.generate.assert_not_called()

    def test_bundle_allowlist_hashes_and_stage0_preservation(self) -> None:
        """Verify that bundle allowlist hashes and stage0 preservation.

        Returns:
            None: This operation returns no value.
        """
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "TEMP") as temporary:
            output = Path(temporary)
            stage0 = output / "ielts_stage0_space.zip"
            stage0.write_bytes(b"preserve stage 0")
            folder, path = build_bundle(output)
            expected = {*APP_FILES, *CORE_FILES, "BUILD_MANIFEST.json"}
            with zipfile.ZipFile(path) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(set(archive.namelist()), expected)
                manifest = json.loads(archive.read("BUILD_MANIFEST.json"))
                self.assertEqual(set(manifest["files"]), expected - {"BUILD_MANIFEST.json"})
                for name, fingerprint in manifest["files"].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), fingerprint)
                    self.assertEqual((folder / name).read_bytes(), archive.read(name))
                self.assertEqual(manifest["source_sha256"], hashlib.sha256(
                    json.dumps(manifest["files"], sort_keys=True).encode("utf-8")
                ).hexdigest())
            self.assertEqual(stage0.read_bytes(), b"preserve stage 0")

    def test_unpacked_artifact_runs_shared_judge_without_repo_imports(self) -> None:
        """Verify that unpacked artifact runs shared judge without repo imports.

        Returns:
            None: This operation returns no value.
        """
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "TEMP") as temporary:
            folder, archive_path = build_bundle(Path(temporary))
            unpacked = Path(temporary) / "unpacked"
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(unpacked)
            script = r'''
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from transformers.tokenization_utils_base import BatchEncoding

root = Path.cwd()
sys.path[:0] = [str(root / "code"), str(root / "code/shared")]
from ielts_judge.llm.Judge import Judge
from Probe import run_probe
assert Path(inspect.getfile(Judge)).is_relative_to(root)
response = '{"task_response": 6, "coherence_and_cohesion": 6, "lexical_resource": 6, "grammatical_range_and_accuracy": 6}'
tokenizer = Mock()
tokenizer.init_kwargs = {}
tokenizer.eos_token_id = 2
tokenizer.apply_chat_template.return_value = "test chat"
tokenizer.return_value = BatchEncoding({"input_ids": torch.tensor([[90, 91]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)})
tokenizer.decode.return_value = response
model = SimpleNamespace(device=torch.device("cpu"), generation_config=SimpleNamespace(eos_token_id=2), eval=Mock(), generate=Mock(return_value=torch.tensor([[90, 91, 10, 2]])))
judge = Judge("qwen25_1p5b", model, tokenizer)
result = run_probe(judge, model_key="qwen25_1p5b", task_prompt="New task", essay="New essay", use_synthetic_context=True)
assert result["raw_text"] == response
assert result["generated_tokens"] == 2
assert result["context_mode"] == "synthetic_fixture"
assert model.generate.call_count == 1
print(json.dumps({"artifact_core_smoke": "ok", "model_id": result["model_id"], "stop_reason": result["stop_reason"]}))
'''
            environment = {**os.environ, "PYTHONPATH": ""}
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script], cwd=unpacked, env=environment,
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["artifact_core_smoke"], "ok")
