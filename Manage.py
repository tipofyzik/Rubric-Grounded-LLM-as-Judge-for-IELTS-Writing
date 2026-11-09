"""Install, launch, verify and package the three IELTS application components."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv
import zipfile

ROOT = Path(__file__).resolve().parent
ACCESS = {
    "HF_TOKEN", "HF_SPACE_ID", "SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY",
    "SUPABASE_SECRET_KEY", "HF_ARTIFACT_REPO", "HF_ARTIFACT_REVISION",
    "HF_ARTIFACT_MANIFEST_SHA256", "IELTS_MODEL_KEY",
}
RESEARCH_COMMANDS = {
    "data": "PrepareData.py", "aes": "TrainAes.py", "panda": "ManageCorpus.py",
    "evaluate": "EvaluateModels.py", "bundle-aes": "tools/BuildAesKaggle.py",
    "bundle-panda": "tools/BuildPandaKaggle.py",
    "bundle-evaluation": "tools/BuildEvaluationKaggle.py",
    "bundle-legacy": "notebooks/BuildBundle.py",
}


def environment(env_file: Path | None = None) -> dict[str, str]:
    """Load supported settings without replacing values supplied by the process.

    Args:
        env_file (Path | None): Optional settings file; defaults to the checkout's .env.

    Returns:
        dict[str, str]: Child-process environment with component import paths and tracing disabled.
    """
    values = dict(os.environ)
    settings_path = Path(env_file) if env_file else ROOT / ".env"
    if settings_path.is_file():
        for line in settings_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if separator and key.strip() in ACCESS:
                values.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    values.update(
        PYTHONPATH=os.pathsep.join((str(ROOT / "code"), str(ROOT / "code/shared"))),
        PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", LANGSMITH_TRACING="false",
    )
    return values


def source_archive(destination: Path) -> int:
    """Package maintained code and guides, excluding local inputs and historical reports.

    Args:
        destination (Path): New ZIP filename; an existing archive is never overwritten.

    Returns:
        int: Zero after writing the archive and its output summary.
    """
    destination = Path(destination).resolve()
    filenames = (
        "Manage.py", "RuntimeAssets.py", "README.md", "LICENSE", ".env.example",
        "requirements.txt", "requirements-dev.txt", "pyrightconfig.json",
    )
    files = [ROOT / filename for filename in filenames if (ROOT / filename).is_file()]
    excluded = {"__pycache__", ".cache", "artifacts", ".venv", ".runtime", ".pytest_cache", ".ruff_cache"}
    extensions = {".py", ".md", ".txt", ".yaml", ".json", ".ipynb", ".sql", ".mjs", ".svg", ".png", ".jpg"}
    for folder in ("code", "tests", "docs"):
        for source_path in (ROOT / folder).rglob("*"):
            if source_path.suffix == ".md":
                if folder != "docs" and source_path.name not in {"README.md", "OPERATING.md"}:
                    continue
                if folder == "docs" and source_path.parent == ROOT / "docs" and source_path.name not in {
                    "README.md", "model_evaluation_protocol.md", "model_evaluation_report.md",
                }:
                    continue
            if source_path.is_file() and source_path.suffix in extensions and not excluded.intersection(source_path.relative_to(ROOT).parts):
                files.append(source_path)
    workflow = ROOT / ".github/workflows/demo-checks.yml"
    if workflow.is_file():
        files.append(workflow)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_path in sorted(set(files)):
            archive.write(source_path, source_path.relative_to(ROOT).as_posix())
    print(json.dumps({"source_archive": str(destination), "files": len(set(files))}))
    return 0


class CommandRunner:
    """Run component commands with one explicit interpreter, checkout and environment."""

    def __init__(self, python: Path, values: dict[str, str]) -> None:
        """Keep the selected execution context for all component subprocesses.

        Args:
            python (Path): Interpreter selected by the caller or the project environment.
            values (dict[str, str]): Prepared environment; credentials remain in memory.

        Returns:
            None: This constructor initializes the runner.
        """
        self.python = python
        self.values = values

    def run(self, arguments: list[str]) -> int:
        """Execute a Python command and propagate its exit status.

        Args:
            arguments (list[str]): Script or module invocation after the interpreter.

        Returns:
            int: Child-process exit status.
        """
        return subprocess.call([str(self.python), "-B", *arguments], cwd=ROOT, env=self.values)

    def install(self, target: str, dry_run: bool) -> int:
        """Create an isolated CPU environment for the selected operator component.

        Args:
            target (str): Public, local or research dependency profile.
            dry_run (bool): Print the installation plan without installing packages.

        Returns:
            int: Zero on success, otherwise the failing installer exit status.
        """
        prepared = ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
        requirements = {
            "public": "code/public_demo/requirements-operator.txt",
            "local": "code/local_backup/requirements.txt",
            "research": "code/offline_research/requirements.txt",
        }[target]
        pip_command = [str(prepared), "-m", "pip", "--isolated", "--cache-dir", str(ROOT / ".cache/pip"),
                       "--disable-pip-version-check", "install", "--timeout", "20", "--retries", "1", "--no-input"]
        plan = [
            [*pip_command, "--upgrade", "pip", "--index-url", "https://pypi.org/simple"],
            [*pip_command, "torch==2.10.0+cpu", "--index-url", "https://download.pytorch.org/whl/cpu", "--extra-index-url", "https://pypi.org/simple"],
            [*pip_command, "-r", requirements, "--index-url", "https://pypi.org/simple"],
        ]
        if dry_run:
            print(json.dumps({"venv": str(ROOT / ".venv"), "commands": plan}, indent=2))
            return 0
        if not prepared.is_file():
            venv.create(ROOT / ".venv", with_pip=True)
        for command in plan:
            exit_code = subprocess.call(command, cwd=ROOT, env=self.values)
            if exit_code:
                return exit_code
        return 0

    def test(self, suite: str) -> int:
        """Run selected offline suites without hosted judge calls.

        Args:
            suite (str): Main, legacy, deployment, portable, or all supported suites.

        Returns:
            int: Zero when each requested suite succeeds, otherwise its failure status.
        """
        self.values.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MPLBACKEND="Agg")
        commands = {
            "main": ["-m", "unittest", "discover", "-s", "tests", "-p", "Test*.py", "-v"],
            "legacy": ["code/offline_research/notebooks/RunChecks.py"],
            "deployment": ["code/public_demo/RuntimeChecks.py"], "portable": ["tests/RunPortable.py"],
        }
        for name in ("main", "legacy", "deployment") if suite == "all" else (suite,):
            exit_code = self.run(commands[name])
            if exit_code:
                return exit_code
        return 0


def create_parser() -> argparse.ArgumentParser:
    """Define the project command interface independently of command execution.

    Returns:
        argparse.ArgumentParser: Parser accepting the supported component commands.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Optional local settings file")
    parser.add_argument("--python", type=Path, help="Explicit interpreter; defaults to project .venv")
    commands = parser.add_subparsers(dest="component", required=True)
    commands.add_parser("public", add_help=False, help="Space artifacts/build/status/deploy")
    commands.add_parser("local", add_help=False, help="Local UI, or local score for the CLI")
    research = commands.add_parser("research", add_help=False, help="Research tools")
    research.add_argument("action", choices=tuple(RESEARCH_COMMANDS))
    commands.add_parser("setup", add_help=False, help="Restore compatible assets and model cache")
    install = commands.add_parser("install", help="Install one component's dependencies")
    install.add_argument("target", choices=("public", "local", "research"))
    install.add_argument("--dry-run", action="store_true")
    tests = commands.add_parser("test", help="Run offline regression checks")
    tests.add_argument("--suite", choices=("all", "main", "legacy", "deployment", "portable"), default="all")
    archive = commands.add_parser("source-zip", help="Package maintained source files")
    archive.add_argument("destination", type=Path)
    return parser


def main(command_arguments: list[str] | None = None) -> int:
    """Dispatch an explicit project command through its selected execution context.

    Args:
        command_arguments (list[str] | None): CLI arguments; None reads process arguments.

    Returns:
        int: Command exit status.
    """
    options, remaining = create_parser().parse_known_args(command_arguments)
    prepared = ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    python = options.python.resolve() if options.python else prepared if prepared.is_file() else Path(sys.executable)
    runner = CommandRunner(python, environment(options.env_file))
    if options.component == "install":
        return runner.install(options.target, options.dry_run)
    if options.component == "source-zip":
        return source_archive(options.destination)
    if options.component == "test":
        return runner.test(options.suite)
    if options.component == "setup":
        return runner.run(["RuntimeAssets.py", *remaining])
    if options.component == "public":
        return runner.run(["code/public_demo/Deployment.py", *remaining])
    if options.component == "local":
        scoring = bool(remaining and remaining[0] == "score")
        filename = "Score.py" if scoring else "App.py"
        return runner.run(["code/local_backup/" + filename, *(remaining[1:] if scoring else remaining)])
    if options.component == "research":
        return runner.run(["code/offline_research/" + RESEARCH_COMMANDS[options.action], *remaining])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
