"""Compare the isolated CPU worker with the accepted in-process models."""

import json
from pathlib import Path
import sys

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]
from local_backup.Runtime import configure_environment

def main() -> None:
    """Parse command options and execute this component's command-line operation.

    Returns:
        None: This operation returns no value.
    """
    configure_environment()
    import numpy as np
    import torch
    from ielts_judge.aes.Inference import load_bundle
    from ielts_judge.panda.Corpus import Encoder
    from CpuWorker import CPUWorker

    torch.set_num_threads(2)
    release_path = ROOT / "results/stage3/release.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    bundle = release_path.parent / release["bundle_path"]
    aes = load_bundle(bundle, device="cpu")
    corpus_release = json.loads((ROOT / "results/stage4/release.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / corpus_release["corpus_path"] / "manifest.json").read_text(encoding="utf-8"))
    encoder = Encoder(manifest["embedding"], device="cpu")
    examples = json.loads((ROOT / "code/shared/resources/demo_examples.json").read_text(encoding="utf-8"))["examples"]
    texts = [example["essay"] for example in examples]
    expected_scores = [aes.predict_essay(essay_text)["scores"] for essay_text in texts]
    expected_vectors, expected_lengths = encoder.encode(texts)
    worker = CPUWorker()
    try:
        isolated_aes = worker.load_aes(bundle)
        isolated_encoder = worker.load_encoder(manifest["embedding"])
        actual_scores = [isolated_aes.predict_essay(essay_text)["scores"] for essay_text in texts]
        actual_vectors, actual_lengths = isolated_encoder.encode(texts)
        # An operation error must not desynchronize the pipe or poison the next request.
        try:
            isolated_aes.predict_essay("")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Empty CPU request was accepted")
        assert isolated_aes.predict_essay(texts[0])["scores"] == expected_scores[0]
    finally:
        worker.close()
    assert not worker.process.is_alive()
    cosines = np.sum(expected_vectors * actual_vectors, axis=1)
    report = {"examples": [example["id"] for example in examples], "scores_equal": expected_scores == actual_scores,
              "scores": actual_scores, "lengths_equal": expected_lengths == actual_lengths,
              "vector_cosines": cosines.tolist(), "worker_recovered_after_error": True,
              "worker_closed": not worker.process.is_alive(),
              "max_absolute_vector_difference": float(np.max(np.abs(expected_vectors - actual_vectors)))}
    assert report["scores_equal"] and report["lengths_equal"]
    assert min(cosines) > 0.999999
    target = ROOT / "results/demo/verification/cpu-worker-parity.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
