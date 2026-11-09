"""Restart the accepted hosted build and grade new input without local inference."""

import json
import argparse
from pathlib import Path
import sys
import time

import requests

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from ielts_judge.Access import read_access
from huggingface_hub import HfApi
from Verify import score


def main() -> None:
    """Restart the deployed Space and verify scoring on a new runtime instance.

    Requires a local deployment record and a successful verification for its
    source revision. Refuses to restart if the live revision differs from the
    recorded deployment.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authenticated", action="store_true")
    current_arguments = parser.parse_args()
    deployment = json.loads((ROOT / "results/demo/deployment.json").read_text(encoding="utf-8"))
    previous = []
    for path in (ROOT / "results/demo/verification").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            continue
        result = record.get("result", {})
        versions = result.get("metadata", {}).get("versions", {})
        if result.get("status") == "ok" and versions.get("source_sha256") == deployment["source_sha256"]:
            previous.append((path.stat().st_mtime, versions["runtime_instance"]))
    before = max(previous)[1]
    access = read_access("rubric_panda_aes", root=ROOT)
    api = HfApi(token=access["HF_TOKEN"])
    repository = deployment["space_id"]
    if api.space_info(repository).sha != deployment["revision"]:
        raise RuntimeError("Live deployment changed; inspect before restarting")
    started = time.monotonic()
    initial = api.restart_space(repository)
    states = [{"stage": str(initial.stage), "seconds": 0}]
    deadline = started + 300
    while time.monotonic() < deadline:
        runtime = api.get_space_runtime(repository)
        if str(runtime.stage) != states[-1]["stage"]:
            states.append({"stage": str(runtime.stage), "seconds": time.monotonic() - started})
            print(json.dumps(states[-1]), flush=True)
        if str(runtime.stage) == "RUNNING" and runtime.raw.get("sha") == deployment["revision"]:
            break
        if str(runtime.stage) in {"BUILD_ERROR", "RUNTIME_ERROR"}:
            raise RuntimeError("Hosted restart failed")
        time.sleep(5)
    else:
        raise TimeoutError("Hosted restart did not finish")
    task = "Some people think cities should provide more safe cycling routes. To what extent do you agree?"
    essay = (
        "I agree that cities should provide more safe cycling routes, although these projects should be planned alongside public transport. "
        "Cycling is useful for short journeys, but many people avoid it because busy roads feel dangerous.\n\n"
        "A connected network matters more than a few isolated lanes. If a route stops at a difficult junction, a new cyclist may still choose "
        "to drive. Councils should therefore examine complete journeys between homes, workplaces and schools. Secure parking also helps people "
        "use a bicycle for ordinary tasks rather than only for exercise.\n\n"
        "There are limits to this approach. Older residents and people travelling long distances may need buses or cars. Taking space from a "
        "crowded pavement can create problems for pedestrians. Good planning should protect these groups and use local evidence to decide "
        "where a separate cycle route is most valuable.\n\n"
        "In conclusion, safer routes can make cycling a practical choice for more residents. Investment should create a connected network "
        "while preserving accessible walking routes and reliable public transport."
    )
    session = requests.Session()
    if current_arguments.authenticated:
        session.headers["Authorization"] = "Bearer " + access["HF_TOKEN"]
    evidence = score("https://" + repository.replace("/", "-") + ".hf.space", session, task, essay)
    result = evidence["result"]
    after = result["metadata"]["versions"]["runtime_instance"]
    report = {"revision": deployment["revision"], "source_sha256": deployment["source_sha256"],
              "before_instance": before, "after_instance": after, "states": states,
              "restart_and_request_seconds": time.monotonic() - started,
              "authenticated": current_arguments.authenticated, "local_inference": False, **evidence}
    target = ROOT / "results/demo/verification/restart-and-new-input.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert result["status"] == "ok" and after != before
    assert result["metadata"]["versions"]["source_sha256"] == deployment["source_sha256"]
    assert result["metadata"]["aes"]["status"] == "ok" and result["metadata"]["retrieval"]["used_count"] >= 1
    print(json.dumps({"restart_verified": True, "new_instance": after, "scores": result["scores"],
                      "restart_and_request_seconds": report["restart_and_request_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
