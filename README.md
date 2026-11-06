# IELTS Writing Evaluator

Evaluate an English IELTS Writing Task 2 question and essay with four integer criterion scores. The application combines an essay-only AES model, a retrieved PANDA training reference, a rubric and Qwen2.5-1.5B-Instruct. It is an experimental evaluator; it does not issue an official IELTS result.

**[Open the public demo](https://tipofyzik-ielts-writing-evaluator.hf.space)** · **[Documentation catalogue](docs/README.md)** · **[Architecture](docs/guides/01Architecture.md)** · **[Installation](docs/guides/04Installation.md)**

The interface supports English, Russian and German. Task and Essay stay English. Three original examples can be edited; evaluation runs only after pressing Evaluate. The application returns Task Response, Coherence and Cohesion, Lexical Resource, and Grammatical Range and Accuracy on the integer 0–9 scale.

![Real hosted scoring result with four criterion scores](docs/media/Demo.jpg)

## Components

| Directory | Responsibility | Entry point |
|---|---|---|
| [Public demo](code/public_demo/README.md) | Autonomous HF web application, isolated CPU models, ZeroGPU judge and deployment | `python Manage.py public --help` |
| [Local backup](code/local_backup/README.md) | Local UI/CLI with CPU AES/BGE and remote judge/database | `python Manage.py local --help` |
| [Shared core](code/shared/README.md) | Scoring, prompt/parser, model profiles, runtime artifacts and localization | Imported by the components |
| [Offline research](code/offline_research/README.md) | Preserved preparation, training, generation and evaluation tools | `python Manage.py research evaluate --help` |
| [Tests](tests/README.md) | Regression, portable, notebook and deployment contracts | `python Manage.py test --suite portable` |

Python source filenames use PascalCase; package markers keep Python's conventional `__init__.py` spelling. Each source block has its own README. The research tools are retained for reproducibility and are outside the current demo-completion scope.

## Local startup

Use Python 3.11 for local/operator commands. The generated Space pins its own Python version in its README metadata.

```powershell
python Manage.py install local
# Create .env from .env.example in a new checkout and fill your own settings.
python Manage.py setup --download-assets --warm-cache
python Manage.py local --check
python Manage.py local
```

Open `http://127.0.0.1:7860`. The installer creates an isolated `.venv`; subsequent manager commands select it automatically. Do not overwrite an existing `.env`. Process environment values take precedence. `--check` checks files, installed versions and endpoint availability; it does not perform real model/database inference.

**The source checkout does not contain trained weights, the corpus or datasets.** Download setup requires an accessible compatible release and immutable `HF_ARTIFACT_*` settings. The author's runtime artifacts are private. Alternatively use `python Manage.py setup --artifact-source PATH_TO_RELEASE --warm-cache`. See [installation and artifact access](docs/guides/04Installation.md) before running setup.

## Hosted deployment

The public application runs on HF with Supabase HTTPS/RPC. Visitors need only the public URL. Operators need their configured Space, private compatible artifacts and reference database; no new hosting platform is required.

```powershell
python Manage.py install public
python Manage.py public build
python Manage.py public status
python Manage.py public deploy
```

Build is local. Deploy updates the configured Space and is an explicit publishing operation. Use `--configure-secrets` only when configuring the Space's runtime secrets. A new database requires migrations 004 then 005 and a separate corpus import; application startup does not migrate or ingest. See the [operator guide](docs/guides/05Deployment.md).

## Existing evaluation

The preserved baseline compares AES with six Qwen context setups on the same 1,635 test essays, one repeat and 9,810 Qwen jobs. It found no improvement from the full hybrid:

| System | Macro MAE ↓ | Macro QWK ↑ |
|---|---:|---:|
| AES | 0.979511 | 0.607192 |
| Rubric + PANDA + AES | 0.981040 | 0.606128 |

The full hybrid repeats 6,536 of 6,540 individual AES scores. These are four-criterion macro metrics, not an overall IELTS band. [Protocol](docs/model_evaluation_protocol.md) and [report](docs/model_evaluation_report.md) retain provenance. Repeatability, robustness, tuning and additional model runs are deferred. Engineering acceptance of the demo is separate from research completion.

## Limits and data handling

- One tested judge profile is loaded at startup. A dropdown does not dynamically load another checkpoint.
- AES sees Essay only; its first score is an essay-only proxy for Task Response. AES/BGE record truncation; the judge does not silently trim Task/Essay.
- Source labels include Writing9 site/crowd scores. Questions can overlap between train/test; the used pool is not an untouched holdout.
- PANDA preserves raw training-reference completions, including failed/partial generation, without repair.
- The local backup uses hosted services and requires network access. Platform availability and visitor quotas affect the public demo.
- The application does not persist submitted text/raw judge completions or add them to the reference corpus. Runtime file routes are blocked.

## Verification and source release

```powershell
python Manage.py test --suite portable
python Manage.py test --suite all
python Manage.py source-zip output/ielts-demo-source.zip
```

Portable checks use no private data, pretrained weights or online services. The full suite also requires local dataset/legacy fixtures. Tests do not substitute for a real hosted acceptance run. The [verification guide](docs/guides/07Verification.md) explains the scope and release checklist.

The source archive excludes `.env`, caches, environments, data, weights, results, IDE settings and the local `.gitignore`. The workspace uses a deny-all publication allowlist. GitHub publication is performed by the repository owner after reviewing the changes. See the [engineering verification report](docs/DemoVerification.md) for the checked source/runtime scope.
