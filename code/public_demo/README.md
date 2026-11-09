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

# Hosted demo

This Hugging Face Space runs the Gradio app for IELTS Writing Task 2.

- [`App.py`](App.py) starts Gradio and connects the scoring engine to Supabase.
- [`CpuWorker.py`](CpuWorker.py) estimates scores and prepares the similarity search query.
- [`Runtime.py`](Runtime.py) fetches the model and reference files and connects scoring requests to Qwen on the GPU.

[Demo](https://tipofyzik-ielts-writing-evaluator.hf.space) · [Source](https://github.com/tipofyzik/Rubric-Grounded-LLM-as-Judge-for-IELTS-Writing) · [Deployment](https://github.com/tipofyzik/Rubric-Grounded-LLM-as-Judge-for-IELTS-Writing/blob/main/docs/guides/05Deployment.md)
