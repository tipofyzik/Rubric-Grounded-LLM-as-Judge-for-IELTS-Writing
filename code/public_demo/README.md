---
title: IELTS Writing Task 2
emoji: 📝
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.9.0
python_version: '3.12'
app_file: App.py
models:
  - Qwen/Qwen2.5-1.5B-Instruct
---

# Public Demo

Own the autonomous HF application, isolated CPU model process, private-artifact restoration and explicit operator deployment. App.py starts the service; SourceBuilder.py defines the public file set; Deployment.py performs operator actions. Visitors use the hosted URL.

The interface returns four integer criterion scores, supports EN/RU/DE and does not persist submitted text or raw judge completions. Runtime secrets are configured in Space settings. Platform startup/visitor quota limits apply.

| File | Responsibility |
|---|---|
| [__init__.py](__init__.py) | IELTS public_demo component. |
| [App.py](App.py) | Autonomous IELTS hybrid demo: CPU AES/BGE, Supabase and ZeroGPU Qwen. |
| [Audit.py](Audit.py) | Verify published bytes, private artifacts and blocked application file routes. |
| [CpuWorker.py](CpuWorker.py) | One CPU model process, isolated from ZeroGPU's forked CUDA workers. |
| [Deployment.py](Deployment.py) | Publish private artifacts or the allowlisted Space; credentials stay in memory. |
| [requirements-operator.txt](requirements-operator.txt) | Configuration, dependency or supporting source input. |
| [requirements.txt](requirements.txt) | Configuration, dependency or supporting source input. |
| [Runtime.py](Runtime.py) | Pinned private artifacts and the CPU-to-ZeroGPU judge boundary. |
| [RuntimeChecks.py](RuntimeChecks.py) | Deployment boundaries using the real builder, runtime and installed imports. |
| [SourceBuilder.py](SourceBuilder.py) | Prepare allowlisted public source and private runtime artifacts. |
| [Verify.py](Verify.py) | Exercise the hosted UI endpoints and save content-free acceptance evidence. |
| [VerifyCpu.py](VerifyCpu.py) | Compare the isolated CPU worker with the accepted in-process models. |
| [VerifyFailures.py](VerifyFailures.py) | Controlled dependency failures over real HTTP; no production DB changes. |
| [VerifyLocal.py](VerifyLocal.py) | Real local fallback scoring against the preserved hosted judge endpoint. |
| [VerifyQueue.py](VerifyQueue.py) | Real hosted overflow and independent-session checks with synthetic inputs. |
| [VerifyRestart.py](VerifyRestart.py) | Restart the accepted hosted build and grade new input without local inference. |

[Documentation catalogue](../../docs/README.md)
