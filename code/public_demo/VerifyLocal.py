"""Real local fallback scoring against the preserved hosted judge endpoint."""

import json
from pathlib import Path
import sys

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from local_backup.Runtime import configure_environment, read_access, preflight, RemoteJudge, load_service


if __name__ == "__main__":
    configure_environment()
    values = read_access("rubric_panda_aes")
    ready = preflight(values, "rubric_panda_aes")
    service = load_service(RemoteJudge(values, ready["space"], ready["judge_endpoint"]), values, "rubric_panda_aes")
    example = json.loads((ROOT / "code/shared/resources/demo_examples.json").read_text(encoding="utf-8"))["examples"][0]
    result = service.score(model_key="qwen25_1p5b", task_prompt=example["task_prompt"], essay=example["essay"])
    (ROOT / "results/demo/verification/local-fallback.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "scores": result.get("scores"), "error": result.get("error")}))
    assert result["status"] == "ok"
