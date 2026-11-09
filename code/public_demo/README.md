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

# Demo

The browser demo runs on Hugging Face. When someone submits a question and essay, it looks up a similar scored example in Supabase and asks Qwen for the final scores.

In the source repository, run `python Manage.py public build` to prepare the Space tree under `results/demo/build/public-space/`. The generated tree includes `DEPLOYMENT.json`, the pinned artifact configuration required by the app. [Operator instructions](https://github.com/tipofyzik/Rubric-Grounded-LLM-as-Judge-for-IELTS-Writing/blob/main/code/public_demo/OPERATING.md)

- [`App.py`](App.py) starts the app from the built Space tree.
- [`CpuWorker.py`](CpuWorker.py) makes the first estimate and prepares the essay for search.
- [`Runtime.py`](Runtime.py) gets the files the demo needs and sends scoring requests to Qwen.

[Demo](https://tipofyzik-ielts-writing-evaluator.hf.space) · [Source](https://github.com/tipofyzik/Rubric-Grounded-LLM-as-Judge-for-IELTS-Writing) · [Deployment](https://github.com/tipofyzik/Rubric-Grounded-LLM-as-Judge-for-IELTS-Writing/blob/main/docs/guides/Deploy.md)
