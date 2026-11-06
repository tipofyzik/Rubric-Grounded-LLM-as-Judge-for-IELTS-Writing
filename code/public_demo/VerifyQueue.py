"""Real hosted overflow and independent-session checks with synthetic inputs."""
from typing import Any

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from uuid import uuid4

import requests

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from ielts_judge.Access import read_access
from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from ielts_judge.Prompts import JudgeContext, ReferenceContext, build_judge_messages
from Verify import call


def main() -> None:
    """Parse command options and execute this component's command-line operation.

    Returns:
        None: This operation returns no value.
    """
    access = read_access("rubric_panda_aes", root=ROOT)
    repository = access["HF_SPACE_ID"]
    base = "https://" + repository.replace("/", "-") + ".hf.space"
    examples = json.loads((ROOT / "code/shared/resources/demo_examples.json").read_text(encoding="utf-8"))["examples"]
    focus = ["cost", "access", "time", "families", "workers", "communities"]
    cases = []
    for index in range(6):
        example = examples[index % 3]
        session = requests.Session()
        session.headers["Authorization"] = "Bearer " + access["HF_TOKEN"]
        identity, language = uuid4().hex, ("en", "ru", "de")[index % 3]
        task = example["task_prompt"] + " Consider the effect on " + focus[index] + "."
        call(base, session, "prepare_submission", [language, "qwen25_1p5b", task, example["essay"]], identity)
        cases.append((session, identity, language, task, example["essay"]))
    barrier = threading.Barrier(6)
    started = time.monotonic()

    def execute(case: Any) -> dict[str, Any]:
        """Execute.

        Args:
            case (Any): Case used by this operation.

        Returns:
            dict[str, Any]: Named result fields and metadata produced by this operation.
        """
        session, identity, language, _, _ = case
        events = []
        barrier.wait(timeout=20)
        try:
            output = call(base, session, "evaluate", [None], identity,
                          observer=lambda event: events.append({"type": event.get("msg"), "seconds": time.monotonic() - started}))
            result = next(item for item in output if isinstance(item, dict) and "run_id" in item)
            return {"session": identity, "language": language, "accepted": True, "result": result,
                    "status_message": output[5], "events": events}
        except requests.HTTPError as error:
            return {"session": identity, "language": language, "accepted": False,
                    "http_status": error.response.status_code, "error": error.response.json(), "events": events}

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(execute, cases))
    accepted = [current_record for current_record in results if current_record["accepted"]]
    report = {"submitted": 6, "accepted": len(accepted), "rejected": 6 - len(accepted), "results": results}
    target = ROOT / "results/demo/verification/queue-and-sessions.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    needed = {identity for current_record in accepted for identity in current_record["result"]["metadata"]["retrieval"]["used_reference_ids"]}
    release = json.loads((ROOT / "results/stage4/release.json").read_text(encoding="utf-8"))
    records = {}
    with (ROOT / release["corpus_path"] / "records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["essay_id"] in needed:
                records[record["essay_id"]] = record
    for case, actual in zip(cases, results):
        if not actual["accepted"]:
            continue
        _, _, language, task, essay = case
        result = actual["result"]
        metadata = result["metadata"]
        references = []
        for identity in metadata["retrieval"]["used_reference_ids"]:
            record = records[identity]
            references.append(ReferenceContext(identity, record["task_prompt"],
                              {key: record["criteria"][key]["score"] for key in PUBLIC_CRITERION_IDS},
                              record["insights"], record["essay"]))
        context = JudgeContext(include_rubric=True, references=tuple(references), aes_priors=metadata["aes"]["scores"])
        expected = build_judge_messages(task, essay, context)
        actual["input_binding_verified"] = hashlib.sha256(json.dumps(expected, ensure_ascii=False).encode()).hexdigest() == metadata["prompt"]["sha256"]
        assert actual["input_binding_verified"] and result["status"] == "ok", (result.get("error"), actual["input_binding_verified"])
        expected_text = {"en": "Evaluation complete", "ru": "Оценка завершена", "de": "Bewertung abgeschlossen"}[language]
        actual["language_verified"] = expected_text in actual["status_message"]
        assert actual["language_verified"], actual["status_message"]
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert 1 <= len(accepted) <= 5 and report["rejected"] >= 1
    assert all(current_record["http_status"] == 503 and "queue" in str(current_record["error"]).lower() for current_record in results if not current_record["accepted"])
    print(json.dumps({"submitted": 6, "accepted": len(accepted), "rejected": report["rejected"],
                      "all_input_bindings_verified": True, "languages_verified": True}))


if __name__ == "__main__":
    main()
