"""Executable notebook bootstrap and source syntax, without pip or GPU jobs."""
from typing import Any

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())


def bootstrap() -> Any:
    """Bootstrap.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    notebook = json.loads((ROOT / "code/offline_research/notebooks/AESTraining.ipynb").read_text(encoding="utf-8"))
    scope = {}
    exec(notebook["cells"][1]["source"], scope)
    return scope["prepare_project"]


def unpacked_fixture(root: Any) -> Any:
    """Unpacked fixture.

    Args:
        root (Any): Project root used to resolve relative configuration paths.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    source = root / "input/datasets/example/ielts-project"
    files = {"code/offline_research/TrainAes.py": "print('fixture')", "code/shared/paths.yaml": "datasets: {}",
             "requirements.txt": "PyYAML"}
    for name, contents in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files}
    return source, hashes


class AESKaggleTests(unittest.TestCase):
    """Manage aeskaggle tests through the operations defined by this component."""
    def test_notebook_cells_parse_and_bootstrap_checks_hash_and_archive_paths(self) -> None:
        """Verify that notebook cells parse and bootstrap checks hash and archive paths.

        Returns:
            None: This operation returns no value.
        """
        notebook = json.loads((ROOT / "code/offline_research/notebooks/AESTraining.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
        scope = {}
        exec(notebook["cells"][1]["source"], scope)
        prepare = scope["prepare_project"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "ielts_kaggle_bundle.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("code/offline_research/TrainAes.py", "print('fixture')")
                output.writestr("code/shared/paths.yaml", "datasets: {}")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            project = prepare(root, root / "working", expected_sha256=digest)
            self.assertTrue((project / "code/offline_research/TrainAes.py").is_file())
            second = prepare(root, root / "working", expected_sha256=digest)
            self.assertNotEqual(project, second)
            self.assertTrue(project.is_dir())
            with self.assertRaisesRegex(ValueError, "SHA256"):
                prepare(root, root / "working", expected_sha256="0" * 64)
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("code/shared/paths.yaml", "datasets: {}")
                output.writestr("../escape.py", "unexpected")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                prepare(root, root / "working")
            self.assertFalse((root / "working/escape.py").exists())

    def test_unpacked_nested_dataset_is_found_and_all_expected_files_are_verified(self) -> None:
        """Verify that unpacked nested dataset is found and all expected files are verified.

        Returns:
            None: This operation returns no value.
        """
        prepare = bootstrap()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, hashes = unpacked_fixture(root)
            (source / "dataset-metadata.json").write_text("extra Kaggle metadata")
            project = prepare(root / "input", root / "working", expected_sha256="0" * 64,
                              expected_files=hashes)
            for name in hashes:
                self.assertEqual((project / name).read_bytes(), (source / name).read_bytes())
            self.assertFalse((project / "dataset-metadata.json").exists())
            self.assertTrue((source / "code/offline_research/TrainAes.py").is_file())

    def test_unpacked_corruption_missing_files_and_missing_hashes_fail(self) -> None:
        """Verify that unpacked corruption missing files and missing hashes fail.

        Returns:
            None: This operation returns no value.
        """
        prepare = bootstrap()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, hashes = unpacked_fixture(root)
            with self.assertRaisesRegex(ValueError, "expected_files"):
                prepare(root / "input", root / "working")
            entry = source / "code/offline_research/TrainAes.py"
            original = entry.read_bytes()
            entry.write_text("altered code")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                prepare(root / "input", root / "working", expected_files=hashes)
            entry.write_bytes(original)
            (source / "requirements.txt").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                prepare(root / "input", root / "working", expected_files=hashes)
            (source / "requirements.txt").write_text("PyYAML")
            prepare(root / "input", root / "working", expected_files=hashes)

    def test_multiple_projects_require_explicit_path_and_input_paths_cannot_escape(self) -> None:
        """Verify that multiple projects require explicit path and input paths cannot escape.

        Returns:
            None: This operation returns no value.
        """
        prepare = bootstrap()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, hashes = unpacked_fixture(root)
            other = root / "input/other/code/shared"
            other.mkdir(parents=True)
            (other / "paths.yaml").write_text("datasets: {}")
            with self.assertRaisesRegex(ValueError, "Expected one"):
                prepare(root / "input", root / "working", expected_files=hashes)
            prepare(root / "input", root / "working", bundle_path=source, expected_files=hashes)
            with self.assertRaisesRegex(ValueError, "Unsafe input path"):
                prepare(root / "input", root / "working", bundle_path=source,
                        expected_files={**hashes, "../escape.py": "unused"})
            self.assertFalse((root / "working/escape.py").exists())

    def test_absent_input_explains_add_input_and_renamed_zip_is_supported(self) -> None:
        """Verify that absent input explains add input and renamed zip is supported.

        Returns:
            None: This operation returns no value.
        """
        prepare = bootstrap()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            input_root.mkdir()
            with self.assertRaisesRegex(ValueError, "Add Input"):
                prepare(input_root, root / "working")
            archive = input_root / "renamed-upload.zip"
            files = {"code/shared/paths.yaml": "datasets: {}", "code/offline_research/TrainAes.py": "print('fixture')"}
            hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()}
            with zipfile.ZipFile(archive, "w") as output:
                for name, value in files.items():
                    output.writestr(name, value)
            project = prepare(input_root, root / "working",
                              expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                              expected_files=hashes)
            self.assertTrue((project / "code/offline_research/TrainAes.py").is_file())


if __name__ == "__main__":
    unittest.main()
