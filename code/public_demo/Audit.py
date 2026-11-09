"""Verify published bytes, private artifacts and blocked application file routes."""

import json
from pathlib import Path
import sys
import zipfile

import requests

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from ielts_judge.Access import read_access
from huggingface_hub import HfApi, snapshot_download
from Runtime import file_hash


def main() -> None:
    """Parse command options and execute this component's command-line operation.

    Returns:
        None: This operation returns no value.
    """
    deployment = json.loads((ROOT / "results/demo/deployment.json").read_text(encoding="utf-8"))
    values = read_access("rubric_panda_aes", root=ROOT)
    api = HfApi(token=values["HF_TOKEN"])
    info = api.space_info(deployment["space_id"])
    assert info.sha == deployment["revision"] and info.private is False
    names = {current_field.rfilename for current_field in info.siblings}
    expected = set(deployment["files"]) | {"DEPLOYMENT.json", ".gitignore"}
    assert names == expected | {".gitattributes"}
    assert not any(name.endswith((".pt", ".bin", ".safetensors", ".jsonl", ".env")) for name in names)
    assert api.model_info(deployment["artifact_repo"], revision=deployment["artifact_revision"]).private is True
    destination = ROOT / "results/demo/published/source"
    snapshot_download(deployment["space_id"], repo_type="space", revision=info.sha,
                      token=values["HF_TOKEN"], local_dir=destination)
    for name, digest in deployment["files"].items():
        assert file_hash(destination / name) == digest, name
    base = "https://" + deployment["space_id"].replace("/", "-") + ".hf.space"
    artifact_root = Path(snapshot_download(
        deployment["artifact_repo"], repo_type="model", revision=deployment["artifact_revision"],
        token=values["HF_TOKEN"], allow_patterns=["ARTIFACT_MANIFEST.json"],
        local_dir=destination.parent / "artifacts",
    ))
    artifact_manifest = artifact_root / "ARTIFACT_MANIFEST.json"
    assert file_hash(artifact_manifest) == deployment["artifact_manifest_sha256"]
    artifact_files = json.loads(artifact_manifest.read_text(encoding="utf-8"))["files"]
    protected_files = [name for name in artifact_files if name.endswith(("/ordinal_model.pt", "/records.jsonl"))]
    assert len(protected_files) == 2
    checks = []
    urls = [
        "https://huggingface.co/" + deployment["artifact_repo"] + "/resolve/" + deployment["artifact_revision"] + "/ARTIFACT_MANIFEST.json",
        *(base + "/gradio_api/file=/app/.runtime/artifacts/" + name for name in protected_files),
        base + "/gradio_api/file=/home/user/app/.runtime/artifacts/ARTIFACT_MANIFEST.json",
        base + "/gradio_api/file=/app/DEPLOYMENT.json",
    ]
    for url in urls:
        with requests.get(url, timeout=20, stream=True) as response:
            checks.append({"url": url, "anonymous_http_status": response.status_code})
            assert response.status_code in (401, 403, 404)
    archive = destination.parent / "public-runtime.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name in sorted(expected):
            output.write(destination / name, name)
    with zipfile.ZipFile(archive) as output:
        assert output.testzip() is None
        assert set(output.namelist()) == expected
    report = {"space_id": info.id, "revision": info.sha, "source_files_verified": len(deployment["files"]),
              "public_repository": True, "private_artifact_repository": True,
              "hardware": info.runtime.raw["hardware"], "anonymous_file_checks": checks,
              "source_archive": str(archive.relative_to(ROOT)), "source_archive_sha256": file_hash(archive)}
    target = ROOT / "results/demo/verification/publication.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_files_verified": report["source_files_verified"], "private_artifacts": True,
                      "file_routes_blocked": True, "hardware": report["hardware"]}))


if __name__ == "__main__":
    main()
