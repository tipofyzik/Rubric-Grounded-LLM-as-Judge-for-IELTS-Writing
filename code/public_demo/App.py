"""Autonomous IELTS hybrid demo: CPU AES/BGE, Supabase and ZeroGPU Qwen."""
from typing import Any

import os
from pathlib import Path
import time
from uuid import uuid4

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / ".runtime/huggingface")
os.environ["GRADIO_TEMP_DIR"] = str(ROOT / ".runtime/gradio")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DO_NOT_TRACK"] = "1"

def main() -> None:
    # A spawned CPU process imports this module without initializing ZeroGPU.
    # spaces must patch torch before model libraries are imported in the parent.
    """Restore pinned runtime inputs and launch the autonomous ZeroGPU scoring interface.

    Returns:
        None: This operation returns no value.
    """
    import spaces
    import gradio as gr
    import torch
    import json
    import resource
    import sys

    sys.path[:0] = [str(ROOT / "code/shared"), str(ROOT / "code/shared")]
    from ielts_judge.llm.Judge import Judge
    from ielts_judge.Paths import find_release
    from ielts_judge.ScoringRuntime import load_scoring_service
    from ielts_judge.Ui import build_demo
    from CpuWorker import CPUWorker
    from Runtime import HostedJudge, restore_artifacts, validate_payload

    deployment = json.loads((ROOT / "DEPLOYMENT.json").read_text(encoding="utf-8"))
    artifacts = restore_artifacts(ROOT, deployment)
    torch.set_num_threads(2)
    judge = Judge.from_pretrained(os.environ.get("IELTS_MODEL_KEY", "qwen25_1p5b"), device="cuda")

    @spaces.GPU(duration=20)
    def generate(payload: Any) -> Any:
        """Generate.

        Args:
            payload (Any): Payload used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        deadline = validate_payload(payload, judge.profile.key)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        result = judge.predict_messages(model_key=payload["model_key"], messages=payload["messages"], deadline=deadline)
        torch.cuda.synchronize()
        result.update(gpu_work_seconds=time.monotonic() - tick,
                      gpu_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                      worker_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        return result

    cpu_worker = CPUWorker()
    service = load_scoring_service(
        HostedJudge(judge, generate), cpu_worker=cpu_worker,
        aes_release=find_release(artifacts, "aes"),
        corpus_release=find_release(artifacts, "panda"), project_root=artifacts,
        supabase_url=os.environ["SUPABASE_URL"], supabase_key=os.environ["SUPABASE_PUBLISHABLE_KEY"],
    )
    service.versions.update(runtime_instance=uuid4().hex, cpu_runtime="spawned_process",
                            source_sha256=deployment["source_sha256"],
                            artifact_revision=deployment["artifact_revision"],
                            startup_seconds=time.monotonic() - STARTED,
                            startup_parent_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    demo = build_demo(service)

    # The local app shares the same admission limit as browser requests.
    with demo:
        judge_payload = gr.JSON(visible=False)
        judge_result = gr.JSON(visible=False)
        judge_button = gr.Button(visible=False)
        judge_button.click(generate, inputs=judge_payload, outputs=judge_result,
                            api_name="judge", api_visibility="undocumented",
                            concurrency_limit=1, concurrency_id="scoring")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                show_error=False, enable_monitoring=False, state_session_capacity=32,
                blocked_paths=[str(ROOT)], footer_links=[])


if __name__ == "__main__":
    main()
