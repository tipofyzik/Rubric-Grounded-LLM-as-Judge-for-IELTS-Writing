"""Real tiny Transformers training, bundles and the full-train consumer contract."""
from typing import Any

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast, RobertaConfig, RobertaModel

ROOT = next(candidate_path for candidate_path in Path(__file__).resolve().parents if (candidate_path / "code/shared/paths.yaml").is_file())
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/shared")]

from ielts_judge.aes.OrdinalRegression import OrdinalHead, OrdinalRegression, token_lengths
from ielts_judge.aes.Inference import read_manifest, write_json
from offline_research.aes.Training import aes_options, metrics_report, run_aes
from ielts_judge.Criteria import PUBLIC_CRITERION_IDS
from ielts_judge.data.Experiment import prepare_experiment
from ielts_judge.data.Preparation import file_hash, text_hash


def tiny_model(max_length: int=16) -> Any:
    """Tiny model.

    Args:
        max_length (int): Max length used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    vocabulary = {word: current_index for current_index, word in enumerate(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "essay", "train", "test", "alpha", "beta", "new"])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="[CLS] $A [SEP]",
                                                 special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                        unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
                                        model_max_length=512)
    encoder = BertModel(BertConfig(vocab_size=len(vocabulary), hidden_size=16,
                                   num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                                   max_position_embeddings=512, pad_token_id=0))
    return OrdinalRegression(encoder=encoder, tokenizer=tokenizer, encoder_name="local-tiny-bert",
                             encoder_revision="0" * 40, max_length=max_length, learning_rate=0.01,
                             device="cpu")


def frame(split: Any, count: int=3) -> Any:
    """Frame.

    Args:
        split (Any): Split used by this operation.
        count (int): Count used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    rows = []
    for current_index in range(count):
        essay = f"essay {split} alpha {current_index}"
        rows.append({"essay_id": f"writing9:{split}:{current_index}", "essay": essay, "task_prompt": f"Task {current_index}",
                     "essay_hash": text_hash(essay), "dataset_source": "writing9", "split": split,
                     "status": "accepted", **{name: (current_index + column_index) % 10 for column_index, name in enumerate(PUBLIC_CRITERION_IDS)}})
    result = pd.DataFrame(rows)
    result.attrs["manifest_sha256"] = "a" * 64
    return result


class AESTests(unittest.TestCase):
    """Manage aestests through the operations defined by this component."""
    @classmethod
    def setUpClass(cls) -> None:
        """SetUpClass.

        Returns:
            None: This operation returns no value.
        """
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        """TearDownClass.

        Returns:
            None: This operation returns no value.
        """
        torch.set_num_threads(cls.threads)

    def setUp(self) -> None:
        """SetUp.

        Returns:
            None: This operation returns no value.
        """
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def trained(self) -> Any:
        """Trained.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        model = tiny_model()
        labels = pd.DataFrame([[0, 1, 8, 9], [9, 8, 1, 0]], columns=PUBLIC_CRITERION_IDS)
        targets = model.ordinal_encode_multi(labels, {current_index: current_index for current_index in range(10)}, 10)
        model.fit(["essay alpha", "essay beta"], targets, epochs=1, batch_size=2)
        return model

    def bundle(self) -> Any:
        """Bundle.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        model = self.trained()
        path = self.root / "bundle"
        model.save_bundle(path, provenance={"purpose": "random tiny encoder contract test"})
        return model, path

    def test_all_integer_classes_round_trip_in_dataset_order(self) -> None:
        """Verify that all integer classes round trip in dataset order.

        Returns:
            None: This operation returns no value.
        """
        values = pd.DataFrame([[current_index] * 4 for current_index in range(10)], columns=PUBLIC_CRITERION_IDS)
        targets = OrdinalRegression.ordinal_encode_multi(values, {current_index: current_index for current_index in range(10)}, 10)
        result = OrdinalRegression.ordinal_decode_multi(torch.tensor(targets * 20 - 10), {current_index: current_index for current_index in range(10)})
        self.assertEqual(result, values.to_numpy().tolist())
        self.assertTrue(all(type(score) is int for row in result for score in row))
        self.assertEqual(OrdinalRegression.ordinal_decode_multi(torch.zeros(1, 4, 9), {current_index:current_index for current_index in range(10)}), [[0]*4])

    def test_fractional_missing_boolean_and_reordered_labels_fail(self) -> None:
        """Verify that fractional missing boolean and reordered labels fail.

        Returns:
            None: This operation returns no value.
        """
        for value in (6.5, -1, 10, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                OrdinalRegression.ordinal_encode([value], {current_index: current_index for current_index in range(10)}, 10)
        with self.assertRaisesRegex(ValueError, "order"):
            OrdinalRegression.ordinal_encode_multi(frame("train")[list(reversed(PUBLIC_CRITERION_IDS))],
                                                    {current_index: current_index for current_index in range(10)}, 10)
        with self.assertRaises(ValueError):
            OrdinalRegression.ordinal_decode_multi(torch.full((1, 4, 9), float("nan")), {current_index:current_index for current_index in range(10)})

    def test_real_backward_updates_encoder_and_heads(self) -> None:
        """Verify that real backward updates encoder and heads.

        Returns:
            None: This operation returns no value.
        """
        model = tiny_model()
        embeddings = cast(torch.nn.Embedding, model.model.encoder.get_input_embeddings())
        first_layer = cast(torch.nn.Linear, cast(OrdinalHead, model.model.heads[0]).net[0])
        encoder_before = embeddings.weight.detach().clone()
        head_before = first_layer.weight.detach().clone()
        targets = np.zeros((2, 4, 9), dtype=np.float32)
        model.fit(["essay alpha", "essay beta"], targets, epochs=1, batch_size=2)
        self.assertFalse(torch.equal(encoder_before, embeddings.weight))
        self.assertFalse(torch.equal(head_before, first_layer.weight))
        self.assertEqual(model.training_metadata["train_count"], 2)
        self.assertEqual(model.predict_logits(["new essay"]).shape, (1, 4, 9))

    def test_length_512_truncation_metadata_and_no_labels_in_predict(self) -> None:
        """Verify that length 512 truncation metadata and no labels in predict.

        Returns:
            None: This operation returns no value.
        """
        model = tiny_model(max_length=512)
        long = "essay " * 600
        result = model.predict_essay(long)
        self.assertEqual(result["input"], {"original_tokens": 602, "used_tokens": 512,
                                          "max_length": 512, "truncated": True})
        self.assertFalse(model.predict_essay("essay " * 510)["input"]["truncated"])
        self.assertEqual(result["scores"], model.predict_essay("essay " * 510)["scores"])
        self.assertEqual(list(result["scores"]), list(PUBLIC_CRITERION_IDS))
        self.assertIn("not task-aware", result["semantics"])
        self.assertEqual(model.predict([]), [])

    def test_invalid_texts_and_impossible_position_limit_fail(self) -> None:
        """Verify that invalid texts and impossible position limit fail.

        Returns:
            None: This operation returns no value.
        """
        model = tiny_model()
        for texts in ([""], [None], "essay"):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                model.predict(texts)
        with self.assertRaisesRegex(ValueError, "position limit"):
            tiny_model(max_length=768)
        encoder = RobertaModel(RobertaConfig(vocab_size=10, hidden_size=16, num_hidden_layers=1,
                                             num_attention_heads=2, intermediate_size=32,
                                             max_position_embeddings=514, pad_token_id=1))
        with self.assertRaisesRegex(ValueError, "position limit"):
            OrdinalRegression(encoder=encoder, tokenizer=model.tokenizer, max_length=513)

    def test_real_bundle_roundtrip_logits_batching_and_tokenizer(self) -> None:
        """Verify that real bundle roundtrip logits batching and tokenizer.

        Returns:
            None: This operation returns no value.
        """
        original, path = self.bundle()
        before = original.predict_logits(["new essay", "essay beta"], batch_size=2)
        with patch("transformers.AutoModel.from_pretrained", side_effect=AssertionError("network forbidden")):
            loaded = OrdinalRegression.load_bundle(path, device="cpu")
        after = loaded.predict_logits(["new essay", "essay beta"], batch_size=1)
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-7)
        self.assertEqual(original.predict(["new essay"]), loaded.predict(["new essay"]))
        self.assertEqual(original.tokenizer("new essay"), loaded.tokenizer("new essay"))
        assert loaded.bundle_manifest is not None
        self.assertEqual(loaded.bundle_manifest["criteria"], list(PUBLIC_CRITERION_IDS))
        self.assertNotIn("optimizer", vars(loaded))
        self.assertFalse(loaded.model.training)

    def test_fresh_offline_process_predicts_from_bundle(self) -> None:
        """Verify that fresh offline process predicts from bundle.

        Returns:
            None: This operation returns no value.
        """
        original, path = self.bundle()
        essay = self.root / "essay.txt"
        essay.write_text("new essay", encoding="utf-8")
        result = subprocess.run([sys.executable, "-B", str(ROOT / "code/offline_research/TrainAes.py"), "predict",
                                 "--bundle", str(path), "--essay-file", str(essay)],
                                cwd=self.root, env={**os.environ, "HF_HUB_OFFLINE": "1"},
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), original.predict_essay("new essay"))

    def test_corrupt_missing_extra_files_and_unsupported_manifest_fail(self) -> None:
        """Verify that corrupt missing extra files and unsupported manifest fail.

        Returns:
            None: This operation returns no value.
        """
        _, path = self.bundle()
        original = json.loads((path / "manifest.json").read_text())
        for key, bad in (("criteria", list(reversed(PUBLIC_CRITERION_IDS))),
                         ("bands", [current_index / 2 for current_index in range(19)]), ("decoding", {}),
                         ("input", {}), ("input", None), ("training", None),
                         ("training", {"status": "untrained"})):
            broken = deepcopy(original)
            broken[key] = bad
            write_json(path / "manifest.json", broken)
            with self.subTest(key=key), self.assertRaises(ValueError):
                OrdinalRegression.load_bundle(path)
        write_json(path / "manifest.json", original)
        extra = path / "unhashed.txt"
        extra.write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "inventory"):
            read_manifest(path)
        extra.unlink()
        weights = path / "ordinal_model.pt"
        good = weights.read_bytes()
        weights.write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "integrity"):
            OrdinalRegression.load_bundle(path)
        weights.write_bytes(good)
        OrdinalRegression.load_bundle(path)  # A valid operation still works after rejection.
        weights.unlink()
        with self.assertRaises(ValueError):
            OrdinalRegression.load_bundle(path)
        with self.assertRaises(FileNotFoundError):
            OrdinalRegression.load_bundle(self.root / "missing")

    def test_wrong_state_shapes_rejected_even_with_updated_file_hash(self) -> None:
        """Verify that wrong state shapes rejected even with updated file hash.

        Returns:
            None: This operation returns no value.
        """
        _, path = self.bundle()
        torch.save({"fc.0.weight": torch.zeros(128, 768)}, path / "ordinal_model.pt")
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["artifacts"]["ordinal_model.pt"] = file_hash(path / "ordinal_model.pt")
        write_json(path / "manifest.json", manifest)
        with self.assertRaises(RuntimeError):
            OrdinalRegression.load_bundle(path)

    def test_untrained_bundle_is_not_accepted_as_trained_and_no_overwrite(self) -> None:
        """Verify that untrained bundle is not accepted as trained and no overwrite.

        Returns:
            None: This operation returns no value.
        """
        model = tiny_model()
        path = self.root / "bundle"
        model.save_bundle(path)
        with self.assertRaisesRegex(ValueError, "completed training"):
            OrdinalRegression.load_bundle(path)
        with self.assertRaises(FileExistsError):
            model.save_bundle(path)

    def test_real_training_pipeline_keeps_full_train_and_aligned_test(self) -> None:
        """Verify that real training pipeline keeps full train and aligned test.

        Returns:
            None: This operation returns no value.
        """
        train, pool = frame("train", 4), frame("test", 5)
        selected, experiment = prepare_experiment(train, pool, config={"sample_size": 3})
        run = self.root / "run"
        with patch("offline_research.aes.Training.OrdinalRegression", wraps=OrdinalRegression) as constructor:
            constructor.side_effect = lambda **keyword_arguments: tiny_model()
            constructor.load_bundle.side_effect = OrdinalRegression.load_bundle
            report = run_aes(train, selected, experiment, run, config={"epochs": 1, "batch_size": 2})
        manifest = read_manifest(run / "aes_bundle")
        self.assertEqual(manifest["training"]["train_count"], 4)
        self.assertEqual(manifest["provenance"]["train"], experiment["train"])
        table = pd.read_csv(run / "aes_predictions.csv")
        self.assertEqual(table.essay_id.tolist(), selected.essay_id.tolist())
        self.assertEqual(report["valid_count"], 3)
        with patch("offline_research.aes.Training.OrdinalRegression") as constructor:
            with self.assertRaises(ValueError):
                run_aes(train.iloc[:2], selected, experiment, self.root / "bad")
            constructor.assert_not_called()
        repeated = run_aes(train, selected, experiment, self.root / "inference",
                           config={"bundle_path": str(run / "aes_bundle")}, train_model=False)
        self.assertIsNone(repeated["fit_seconds"])
        with self.assertRaisesRegex(ValueError, "bundle_path"):
            run_aes(train, selected, experiment, self.root / "missing-inference", train_model=False)

    def test_metrics_undefined_policy_and_options(self) -> None:
        """Verify that metrics undefined policy and options.

        Returns:
            None: This operation returns no value.
        """
        labels = np.zeros((3, 4), dtype=int)
        report = metrics_report(labels, labels, labels)
        self.assertIsNone(report[PUBLIC_CRITERION_IDS[0]]["qwk"])
        self.assertEqual(report[PUBLIC_CRITERION_IDS[0]]["baseline_mae"], 0)
        for options in ({"epochs": True}, {"batch_size": 0}, {"typo": 1}):
            with self.assertRaises(ValueError):
                aes_options(options)


if __name__ == "__main__":
    unittest.main()
