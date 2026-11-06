"""Exercise the hosted UI endpoints and save content-free acceptance evidence."""
from typing import Any

import argparse
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import requests

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from ielts_judge.Access import read_access


def call(base: Any, session: Any, name: Any, data: Any, session_hash: Any, observer: Any=None) -> Any:
    """Call.

    Args:
        base (Any): Base used by this operation.
        session (Any): Session used by this operation.
        name (Any): Name used by this operation.
        data (Any): Data used by this operation.
        session_hash (Any): Session hash used by this operation.
        observer (Any): Observer used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    config = session.get(base + "/config", timeout=20)
    config.raise_for_status()
    function = next(current_record["id"] for current_record in config.json()["dependencies"] if current_record.get("api_name") == name)
    response = session.post(base + "/gradio_api/queue/join",
                            json={"data": data, "session_hash": session_hash, "fn_index": function}, timeout=20)
    response.raise_for_status()
    event_id = response.json()["event_id"]
    with session.get(base + "/gradio_api/queue/data", params={"session_hash": session_hash},
                     stream=True, timeout=(15, 140)) as stream:
        stream.raise_for_status()
        for line in stream.iter_lines(chunk_size=1, decode_unicode=True):
            if line.startswith("data:"):
                message = json.loads(line[5:].strip())
                if observer is not None:
                    observer(message)
                if message.get("msg") == "process_completed" and message.get("event_id") == event_id:
                    if not message["success"]:
                        raise RuntimeError("Hosted endpoint error: " + str(message["output"].get("error")))
                    return message["output"]["data"]
    raise RuntimeError("Hosted response ended without a result")


def score(base: Any, session: Any, task: Any, essay: str, language: str="en") -> dict[str, Any]:
    """Evaluate one Task/Essay submission and return a bounded success or failure result.

    Args:
        base (Any): Base used by this operation.
        session (Any): Session used by this operation.
        task (Any): Task used by this operation.
        essay (str): English essay text supplied for evaluation.
        language (str): Language used by this operation.

    Returns:
        dict[str, Any]: Status, four validated criterion scores or an error, and safe processing metadata.
    """
    identity = uuid4().hex
    started = time.monotonic()
    call(base, session, "prepare_submission", [language, "qwen25_1p5b", task, essay], identity)
    outputs = call(base, session, "evaluate", [None], identity)
    result = next((item for item in outputs if isinstance(item, dict) and "run_id" in item), None)
    if result is None:
        raise RuntimeError("Hosted UI did not return a service result")
    return {"elapsed_including_queue_seconds": time.monotonic() - started, "result": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=["transport", "education", "remote_work", "empty", "long", "own"])
    parser.add_argument("--anonymous", action="store_true")
    current_arguments = parser.parse_args()
    values = read_access("rubric_panda_aes", root=ROOT)
    repository = values["HF_SPACE_ID"]
    base = "https://" + repository.replace("/", "-") + ".hf.space"
    session = requests.Session()
    if not current_arguments.anonymous:
        session.headers["Authorization"] = "Bearer " + values["HF_TOKEN"]
    examples = json.loads((ROOT / "code/shared/resources/demo_examples.json").read_text(encoding="utf-8"))["examples"]
    if current_arguments.scenario in {"empty", "long"}:
        task, essay = "Discuss public libraries.", "" if current_arguments.scenario == "empty" else "Library " * 6000
    elif current_arguments.scenario == "own":
        task = "Some people believe public libraries are no longer necessary because information is available online. To what extent do you agree or disagree?"
        essay = (
            "The internet has changed how people find information, but I disagree that public libraries have become unnecessary. "
            "Libraries provide access to resources and places to study that are not available equally to everyone.\n\n"
            "First, having information online does not mean every household can use it. Some families cannot afford a reliable "
            "computer or a quiet room. A public library gives students a place to complete assignments and gives adults a way "
            "to apply for jobs. Staff can also help visitors judge whether an unfamiliar website is trustworthy. These practical "
            "services matter even when many books have digital editions.\n\n"
            "Second, libraries serve as local meeting places. Reading groups, language classes and activities for children "
            "allow residents to learn together. Someone who feels isolated may benefit from a regular activity that costs "
            "nothing to attend. Online services can support these activities, but they do not always replace direct contact.\n\n"
            "However, libraries should adapt their spending. A building with outdated opening hours and little useful material "
            "will attract few visitors. Councils should consult residents, provide digital access and maintain collections "
            "that reflect local needs. Sharing some services between nearby branches may help control costs.\n\n"
            "In conclusion, libraries remain valuable because access, guidance and community space are different from the "
            "mere availability of information. Updating their services is a better response to technological change than closing them."
        )
    else:
        example = next(example for example in examples if example["id"] == current_arguments.scenario)
        task, essay = example["task_prompt"], example["essay"]
    evidence = score(base, session, task, essay)
    evidence.update(scenario=current_arguments.scenario, authenticated=not current_arguments.anonymous)
    directory = ROOT / "results/demo/verification"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (current_arguments.scenario + ("-anonymous" if current_arguments.anonymous else "")
                        + "-" + evidence["result"]["run_id"] + ".json")
    path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    result = evidence["result"]
    print(json.dumps({"scenario": current_arguments.scenario, "status": result["status"], "scores": result.get("scores"),
                      "error": result.get("error"), "judge_attempts": result["metadata"]["judge_attempts"],
                      "elapsed": evidence["elapsed_including_queue_seconds"], "saved": str(path)}))
    if current_arguments.scenario in {"empty", "long"}:
        assert result["status"] == "error" and result["metadata"]["judge_attempts"] == 0
    else:
        assert result["status"] == "ok", result.get("error")
        assert result["metadata"]["aes"]["status"] == "ok"
        assert result["metadata"]["retrieval"]["used_count"] >= 1
