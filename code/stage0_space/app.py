"""Infrastructure probe for a Gradio ZeroGPU Space; no IELTS scoring."""

import time

STARTED_AT = time.perf_counter()

import spaces
import gradio as gr
import importlib.metadata
import os
import resource

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TASK = "Some people think schools should teach practical skills alongside academic subjects. To what extent do you agree?"
ESSAY = """Schools should teach practical skills alongside academic subjects because students need both kinds of knowledge. Academic subjects help young people understand the world, while practical lessons help them manage everyday responsibilities.

For example, mathematics gives students a foundation for understanding percentages and interest. A lesson about personal budgets can then show how these ideas apply to rent, savings and borrowing. Cooking lessons can similarly connect scientific knowledge about nutrition to decisions students make every day.

However, schools have limited teaching time. Adding many separate subjects could reduce the attention given to reading, mathematics and science. Schools should therefore integrate practical activities into existing lessons where possible. Students might compare household energy costs in mathematics or practise writing a job application in a language class.

Practical education should also recognise that students have different circumstances. A school should provide useful opportunities without assuming that every family can supply equipment or pay for materials. Simple classroom activities can make these lessons accessible.

In conclusion, I agree that practical skills belong in education. Combining them with academic learning can help students use their knowledge, provided that schools plan the activities carefully and preserve time for essential subjects."""

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, dtype=torch.float16,
)
# ZeroGPU optimizes CUDA placement performed at module startup.
model.to("cuda")
model.eval()
STARTUP_SECONDS = time.perf_counter() - STARTED_AT


def environment():
    return {
        "model_id": MODEL_ID,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "startup_seconds": round(STARTUP_SECONDS, 2),
        "process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("gradio", "spaces", "torch", "transformers", "requests", "huggingface-hub")
        },
    }


def check_supabase():
    url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "").strip()
    if not url or not key:
        return {"ok": False, "error": "Set SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY in Space settings."}
    if not url.startswith("https://") or not key.startswith("sb_publishable_"):
        return {"ok": False, "error": "Use an HTTPS Project URL and a publishable key."}
    started = time.perf_counter()
    try:
        with requests.post(
            f"{url}/rest/v1/rpc/stage0_healthcheck",
            headers={"apikey": key},
            json={},
            timeout=(5, 10),
            allow_redirects=False,
        ) as response:
            status = response.status_code
            if status != 200:
                return {"ok": False, "http_status": status}
            result = response.json()
    except (requests.RequestException, ValueError) as error:
        return {"ok": False, "error": type(error).__name__}
    return {
        "ok": result is True,
        "http_status": status,
        "rpc_result": result,
        "seconds": round(time.perf_counter() - started, 3),
    }


@spaces.GPU(duration=60)
def generate():
    messages = [
        {"role": "system", "content": "Briefly explain whether the essay addresses the question, in at most three sentences. Do not assign IELTS scores."},
        {"role": "user", "content": f"Question:\n{TASK}\n\nEssay:\n{ESSAY}"},
    ]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda")
    input_tokens = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    completion = outputs[0, input_tokens:]
    return tokenizer.decode(completion, skip_special_tokens=True), {
        "gpu_work_seconds": round(elapsed, 3),
        "input_tokens": input_tokens,
        "output_tokens": int(completion.numel()),
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "environment": environment(),
    }


with gr.Blocks() as demo:
    gr.Markdown("# IELTS project: stage 0\nInfrastructure checks using a synthetic essay. This is not an IELTS scoring demo.")
    gr.Textbox(value=TASK, label="Synthetic question", interactive=False)
    gr.Textbox(value=ESSAY, label="Synthetic essay", lines=8, interactive=False)
    rpc_button = gr.Button("1. Check Supabase HTTPS/RPC")
    rpc_result = gr.JSON(label="RPC result")
    rpc_button.click(check_supabase, outputs=rpc_result, api_name=False)
    generate_button = gr.Button("2. Generate with Qwen on ZeroGPU")
    completion_output = gr.Textbox(label="Model completion", interactive=False)
    generation_metrics = gr.JSON(label="Generation diagnostics")
    generate_button.click(
        generate, outputs=[completion_output, generation_metrics],
        concurrency_limit=1, concurrency_id="qwen", api_name=False,
    )
    gr.Markdown("Run generation twice, then restart the Space and repeat. GPU work time excludes queue/allocation wait and is not a quota billing measurement.")

demo.queue(max_size=4, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(show_error=False)
