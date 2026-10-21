---
title: IELTS Stage 0
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: '3.10'
app_file: app.py
models:
  - Qwen/Qwen2.5-1.5B-Instruct
---

# Stage 0: Qwen + ZeroGPU + Supabase

This is an infrastructure probe with a synthetic question and essay. It does not implement AES, PANDA retrieval, IELTS scoring, or LangChain. On 2026-10-21 the user reported successful hosted RPC, repeated Qwen generation and another successful RPC/generation after rebuilding. Reported versions and model commit match the pins below. Queue wait and quota usage were not measured; validate the final hybrid usage budget before publication.

## Deploy to the existing private Space

Upload **only these three files** to the Space root: `app.py`, `requirements.txt`, and this `README.md`. The repository containing the research project is not the deployment bundle.

Keep Gradio SDK and ZeroGPU hardware selected. The YAML above selects Gradio 5.49.1 and Python 3.10; it does not allocate GPU hardware. Requirements pin the observed torch 2.10.0, transformers 5.2.0, requests 2.32.5, spaces 0.51.3 and huggingface-hub 1.16.1. Model and tokenizer both use observed commit `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. These pins reproduce the observed direct versions; they are not a complete transitive lockfile. Confirm them by rebuilding and repeating the hosted checks.

In Space **Settings → Variables and secrets**, add:

| Kind | Name | Value |
|---|---|---|
| Variable | `SUPABASE_URL` | Your HTTPS Project URL, without `/rest/v1` |
| Secret | `SUPABASE_PUBLISHABLE_KEY` | Your `sb_publishable_...` key |

The test needs only the publishable key and the `stage0_healthcheck` function already created in Supabase. A database password, secret/service-role key, and HF token are not required. The Qwen repository is public. No credentials belong in uploaded files.

## Verify

1. Wait for the Space to become **Running**. Model download and startup can take several minutes. Build/startup failure is not a passed check; inspect the logs.
2. Click **Check Supabase HTTPS/RPC**. Require `ok: true`, `http_status: 200`, and `rpc_result: true`. This checks a boolean function, not vector retrieval or table permissions. HTTP errors show their status; transport/JSON errors show their type without response bodies or credentials. The connect/read timeouts are not a strict end-to-end deadline.
3. Click **Generate with Qwen on ZeroGPU**. Require a nonempty new completion and diagnostics showing a GPU name, positive output token count, and timings. The output is a short discussion of a synthetic essay, not a score. Generation/queue errors must be reported as failures; no cached result is substituted.
4. Repeat generation to observe a warm request. Save both diagnostic JSON outputs locally. Do not infer model quality from these answers.
5. Restart the Space and repeat RPC and generation. Record startup time, process peak RSS, GPU peak tensor allocation, versions and model commit. Peak RSS is Linux process high-water memory, not current RAM or a measurement of future AES/embedding costs. GPU tensor allocation is not total device memory.
6. Record queue/allocation wait separately and compare HF's remaining quota before/after requests. `gpu_work_seconds` measures work inside the GPU function after allocation; it is not the total user latency or billed quota. Quota and cold-start behavior still need direct observation.

Queue length is limited to four and model concurrency to one. Generation is deterministic-config (`do_sample=False`) with at most 96 new tokens and a 60-second ZeroGPU allocation limit. Repeated outputs may be identical. The model is placed on CUDA at module startup as required by ZeroGPU, not lazily moved inside the decorated call.

Report results in the existing project checklist. The technical platform check passed on the reported runs. Final hybrid latency and quota validation belong to deployment stage 7; do not extrapolate them from this synthetic probe.

## Sources

- [ZeroGPU lifecycle and supported runtime](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Qwen model usage](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Spaces README configuration](https://huggingface.co/docs/hub/spaces-config-reference)
- [Supabase API keys](https://supabase.com/docs/guides/getting-started/api-keys)
