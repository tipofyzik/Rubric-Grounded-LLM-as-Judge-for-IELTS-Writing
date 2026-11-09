"""Publish private artifacts or the allowlisted Space; credentials stay in memory."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from ielts_judge.Access import read_access
from huggingface_hub import HfApi, snapshot_download
from SourceBuilder import prepare_artifacts, prepare_source, write_json
from Runtime import file_hash


def main() -> None:
    """Build, publish or inspect the hosted demo from the command line.

    The build action assembles the Space locally. The artifacts action uploads
    runtime assets to a private repository. The status action reads the Space.
    The deploy action backs up the current revision, optionally sets runtime
    secrets, and replaces Space files outside the new build except .gitattributes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["artifacts", "build", "deploy", "status"])
    parser.add_argument("--configure-secrets", action="store_true",
                        help="Set runtime secrets from the existing local configuration")
    current_arguments = parser.parse_args()
    access = read_access("rubric_panda_aes", root=ROOT) if current_arguments.action != "build" else dict(os.environ)
    api = HfApi(token=access.get("HF_TOKEN"))
    space_id = access.get("HF_SPACE_ID", "")
    if current_arguments.action != "build" and not space_id:
        raise ValueError("Configure your own HF_SPACE_ID (owner/name)")
    artifact_repo = os.environ.get("HF_ARTIFACT_REPO") or space_id.rsplit("/", 1)[0] + "/ielts-writing-runtime"
    work = ROOT / "results/demo"
    artifact_release = work / "artifact-release.json"
    if current_arguments.action == "status":
        info = api.space_info(space_id)
        print(json.dumps({"id": info.id, "sha": info.sha, "private": info.private,
                          "runtime": info.runtime.raw if info.runtime else None}, ensure_ascii=True))
        return
    if current_arguments.action == "artifacts":
        target = work / "build/private-artifacts"
        manifest = prepare_artifacts(target)
        api.create_repo(artifact_repo, private=True, exist_ok=True)
        if api.model_info(artifact_repo).private is not True:
            raise RuntimeError("Artifact repository must be private; upload refused")
        commit = api.upload_folder(repo_id=artifact_repo, folder_path=target,
                                   allow_patterns=[*manifest["files"], "ARTIFACT_MANIFEST.json"],
                                   ignore_patterns=[".cache/**"],
                                   commit_message="Pinned accepted AES and PANDA corpus for hosted inference")
        release = {"artifact_repo": artifact_repo, "artifact_revision": commit.oid,
                   "artifact_manifest_sha256": file_hash(target / "ARTIFACT_MANIFEST.json")}
        write_json(artifact_release, release)
        print(json.dumps(release))
        return
    if artifact_release.is_file():
        release = json.loads(artifact_release.read_text(encoding="utf-8"))
    else:
        names = {"artifact_repo":"HF_ARTIFACT_REPO", "artifact_revision":"HF_ARTIFACT_REVISION", "artifact_manifest_sha256":"HF_ARTIFACT_MANIFEST_SHA256"}
        if any(not os.environ.get(value) for value in names.values()):
            raise ValueError("Publish your compatible artifacts first, or configure immutable HF_ARTIFACT_* settings")
        release = {key:os.environ[value] for key,value in names.items()}
    target = work / "build/public-space"
    deployment = prepare_source(target, **release)
    if current_arguments.action == "build":
        print(json.dumps({"source_sha256": deployment["source_sha256"], "directory": str(target)}))
        return
    info = api.space_info(space_id)
    backup = work / "before" / info.sha
    snapshot_download(space_id, repo_type="space", revision=info.sha,
                      token=access["HF_TOKEN"], local_dir=backup)
    # The existing public ZeroGPU hardware and subscription are left intact.
    if current_arguments.configure_secrets:
        for name in ("HF_TOKEN", "SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY"):
            api.add_space_secret(space_id, key=name, value=access[name])
        api.add_space_secret(space_id, key="IELTS_MODEL_KEY", value=access.get("IELTS_MODEL_KEY", "qwen25_1p5b"))
    # Reconcile Space contents with the build in the same revision-guarded commit.
    current = {candidate_path.rfilename for candidate_path in info.siblings or []}
    public = set(deployment["files"]) | {"DEPLOYMENT.json", ".gitignore"}
    obsolete = sorted(current - public - {".gitattributes"})
    commit = api.upload_folder(repo_id=space_id, repo_type="space", folder_path=target,
                               allow_patterns=sorted(public),
                               delete_patterns=obsolete,
                               commit_message="Autonomous public IELTS hybrid with private runtime artifacts",
                               parent_commit=info.sha)
    write_json(work / "deployment.json", {"space_id": space_id, "revision": commit.oid,
                                         "previous_revision": info.sha, **deployment})
    print(json.dumps({"space_id": space_id, "revision": commit.oid,
                      "previous_revision": info.sha, "deleted_obsolete_files": obsolete}))


if __name__ == "__main__":
    main()
