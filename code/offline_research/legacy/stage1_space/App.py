"""Editable hosted probe for the Stage 1 Judge.predict on ZeroGPU."""
from typing import Any

import time

STARTED_AT = time.perf_counter()

import importlib.metadata
import json
from pathlib import Path
import resource
import sys

import spaces
import gradio as gr
import torch

# The deployment bundle contains the shared package, not another inference engine.
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from ielts_judge.llm.Judge import Judge
from ielts_judge.ModelRegistry import ModelRegistry
from Probe import DEFAULT_TASK, DEFAULT_ESSAY, run_probe


registry = ModelRegistry()
profiles = registry.enabled_profiles()
if len(profiles) != 1:
    raise ValueError("This probe loads exactly one enabled judge profile.")
model_key = profiles[0].key
# Keep the placement lifecycle that was checked by the stage 0 Space.
judge = Judge.from_pretrained(model_key, device="cuda", registry=registry)
STARTUP_SECONDS = time.perf_counter() - STARTED_AT
manifest_path = Path(__file__).resolve().parent / "BUILD_MANIFEST.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))


@spaces.GPU(duration=60)
def evaluate(selected_model: Any, task: Any, essay: str, use_context: bool) -> Any:
    """Evaluate.

    Args:
        selected_model (Any): Selected model used by this operation.
        task (Any): Task used by this operation.
        essay (str): English essay text supplied for evaluation.
        use_context (bool): Use context used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = run_probe(
            judge, model_key=selected_model, task_prompt=task, essay=essay,
            use_synthetic_context=use_context,
        )
    except ValueError as error:
        # Expected input/profile/token-budget rejections have readable UI messages.
        raise gr.Error(str(error)) from error
    torch.cuda.synchronize()
    raw_text = result.pop("raw_text")
    result.update({
        "gpu_work_seconds": round(time.perf_counter() - started, 3),
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "startup_seconds": round(STARTUP_SECONDS, 2),
        "process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "loaded_model_revision": getattr(judge.model.config, "_commit_hash", None),
        "source_sha256": manifest["source_sha256"],
        "packages": {name: importlib.metadata.version(name) for name in (
            "gradio", "spaces", "torch", "transformers", "huggingface-hub", "PyYAML",
        )},
    })
    return raw_text, result


def clear_result() -> Any:
    """Clear result.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    return "", None


with gr.Blocks() as demo:
    gr.Markdown(
        "# Stage 1: real judge smoke test\n"
        "Editable Task 2 input using the shared Judge.predict. "
        "Outputs are raw model completions; score validation and the full hybrid pipeline come later."
    )
    model = gr.Dropdown(
        choices=[(profile.label, profile.key) for profile in profiles],
        value=model_key, label="Judge model",
    )
    task = gr.Textbox(value=DEFAULT_TASK, label="Task / Question", lines=3)
    essay = gr.Textbox(value=DEFAULT_ESSAY, label="Essay", lines=12)
    use_context = gr.Checkbox(value=False, label="Use synthetic context (wiring test only)")
    gr.Markdown("The optional fixture contains artificial scores and a reference. It does not run AES or query Supabase.")
    button = gr.Button("Run Stage 1 judge")
    completion = gr.Textbox(label="Raw completion", lines=8, interactive=False)
    diagnostics = gr.JSON(label="Diagnostics — save after each test")
    for component in (model, task, essay, use_context):
        component.change(clear_result, outputs=[completion, diagnostics], queue=False, api_name=False)
    button.click(
        clear_result, outputs=[completion, diagnostics], queue=False, api_name=False,
    ).then(
        evaluate, inputs=[model, task, essay, use_context], outputs=[completion, diagnostics],
        concurrency_limit=1, concurrency_id="judge", api_name=False,
    )
    gr.Markdown(
        "Each click runs generation. Identical answers are possible with do_sample=False. "
        "GPU work excludes queue/allocation wait and is not a quota billing measurement."
    )

demo.queue(max_size=4, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(show_error=False)
