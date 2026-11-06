"""Permanent project runtime configuration; no credentials or model calls."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from local_backup import Runtime as app_runtime


class AppRuntimeTests(unittest.TestCase):
    """Manage app runtime tests through the operations defined by this component."""
    def setUp(self) -> None:
        """SetUp.

        Returns:
            None: This operation returns no value.
        """
        self.folder = tempfile.TemporaryDirectory(dir=ROOT / ".cache")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.scope = patch.object(app_runtime, "ROOT", self.root)
        self.scope.start()
        self.addCleanup(self.scope.stop)

    def test_cache_is_in_project_and_does_not_use_temp(self) -> None:
        """Verify that cache is in project and does not use temp.

        Returns:
            None: This operation returns no value.
        """
        with patch.dict(os.environ, {}, clear=True):
            app_runtime.configure_environment()
            self.assertEqual(Path(os.environ["HF_HOME"]), self.root / ".cache/huggingface")
            self.assertEqual(Path(os.environ["GRADIO_TEMP_DIR"]), self.root / ".cache/gradio")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_root_env_preserves_runtime_values_and_excludes_import_secret(self) -> None:
        """Verify that root env preserves runtime values and excludes import secret.

        Returns:
            None: This operation returns no value.
        """
        (self.root / ".env").write_text(
            'HF_TOKEN="RUNTIME_ONLY"\nSUPABASE_URL=https://example.invalid\n'
            'SUPABASE_PUBLISHABLE_KEY=PUBLIC_ONLY\nSUPABASE_SECRET_KEY=NEVER_READ\n', encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            values = app_runtime.read_access("rubric_panda_aes")
        self.assertEqual(values["HF_TOKEN"], "RUNTIME_ONLY")
        self.assertEqual(values["SUPABASE_PUBLISHABLE_KEY"], "PUBLIC_ONLY")
        self.assertNotIn("SUPABASE_SECRET_KEY", values)

    def test_environment_variables_work_without_env_file(self) -> None:
        """Verify that environment variables work without env file.

        Returns:
            None: This operation returns no value.
        """
        values = {"HF_TOKEN": "ENV_ONLY", "SUPABASE_URL": "https://example.invalid", "SUPABASE_PUBLISHABLE_KEY": "KEY"}
        with patch.dict(os.environ, values, clear=True):
            self.assertEqual(app_runtime.read_access("rubric_panda_aes"), values)

    def test_environment_overrides_file(self) -> None:
        """Verify that environment overrides file.

        Returns:
            None: This operation returns no value.
        """
        (self.root / ".env").write_text("HF_TOKEN=FILE_ONLY", encoding="utf-8")
        with patch.dict(os.environ, {"HF_TOKEN": "ENV_ONLY"}, clear=True):
            self.assertEqual(app_runtime.read_access("base")["HF_TOKEN"], "ENV_ONLY")

    def test_missing_settings_report_names_without_values_or_legacy_paths(self) -> None:
        """Verify that missing settings report names without values or legacy paths.

        Returns:
            None: This operation returns no value.
        """
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(app_runtime.LauncherError) as error:
            app_runtime.read_access("rubric_panda_aes")
        self.assertEqual(error.exception.code, "missing_access")
        self.assertIn(".env", str(error.exception))
        self.assertNotIn("TEMP", str(error.exception))

    def test_import_failure_names_active_and_project_interpreters(self) -> None:
        """Verify that import failure names active and project interpreters.

        Returns:
            None: This operation returns no value.
        """
        spec = importlib.util.spec_from_file_location("local_ui_app", ROOT / "code/local_backup/App.py")
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        output = io.StringIO()
        with patch.object(app, "read_access", return_value={"HF_TOKEN": "NEVER_PRINT"}), \
             patch.object(app, "preflight", return_value={"space": "https://example.invalid"}), \
             patch.object(app, "RemoteJudge", side_effect=ImportError("Known dependency conflict")), \
             redirect_stdout(io.StringIO()), redirect_stderr(output):
            code = app.main([])
        self.assertEqual(code, 1)
        self.assertIn("Known dependency conflict", output.getvalue())
        self.assertIn(sys.executable, output.getvalue())
        self.assertIn(str(ROOT / ".venv/Scripts/python.exe"), output.getvalue())
        self.assertNotIn("NEVER_PRINT", output.getvalue())

    def test_ide_launch_uses_permanent_environment(self) -> None:
        """Verify that ide launch uses permanent environment.

        Returns:
            None: This operation returns no value.
        """
        config = json.loads((ROOT / ".vscode/launch.json").read_text())["configurations"][0]
        interpreter = Path(config["python"].replace("${workspaceFolder}", str(ROOT)))
        self.assertEqual(interpreter, ROOT / ".venv/Scripts/python.exe")
        self.assertTrue(interpreter.is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
