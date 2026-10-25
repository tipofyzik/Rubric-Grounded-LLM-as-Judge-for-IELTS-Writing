"""Essay-only AES: four ordinal heads and integer scores in dataset order."""

import random
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..criteria import CRITERION_IDS, PUBLIC_CRITERION_IDS

ENCODER_ID = "FacebookAI/roberta-base"
ENCODER_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_texts(texts):
    if not isinstance(texts, list) or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("AES expects a list of nonempty essay strings")


def token_lengths(tokenizer, texts, batch_size=128):
    """Count complete inputs, including special tokens, without truncation."""
    validate_texts(texts)
    positive_integer(batch_size, "batch_size")
    lengths = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], truncation=False,
                            padding=False, verbose=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def length_summary(lengths, max_length):
    return {"count": len(lengths), "truncated_count": sum(n > max_length for n in lengths),
            "max_original_tokens": max(lengths, default=0), "max_length": max_length}


class EssayDataset(Dataset):
    def __init__(self, texts, targets, *, tokenizer, max_length=512):
        validate_texts(texts)
        self.texts, self.targets = texts, targets
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.texts[idx], truncation=True,
                                 padding="max_length", max_length=self.max_length,
                                 return_tensors="pt")
        item = {key: encoded[key].squeeze(0) for key in ("input_ids", "attention_mask")}
        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


class OrdinalHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes - 1),
        )

    def forward(self, x):
        return self.net(x)


class OrdinalRegressionModel(nn.Module):
    def __init__(self, num_classes: int, num_tasks: int, *, encoder: PreTrainedModel):
        super().__init__()
        self.encoder = encoder
        self.num_tasks, self.num_classes = num_tasks, num_classes
        hidden = encoder.config.hidden_size
        self.heads = nn.ModuleList([OrdinalHead(hidden, num_classes) for _ in range(num_tasks)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return torch.stack([head(pooled) for head in self.heads], dim=1)


class OrdinalRegression:
    def __init__(self, num_classes=10, num_tasks=4, device=None, *,
                 encoder_name=ENCODER_ID, encoder_revision=ENCODER_REVISION,
                 encoder: PreTrainedModel | None = None,
                 tokenizer: PreTrainedTokenizerBase | None = None,
                 max_length=512, learning_rate=2e-5, seed=42):
        if type(num_classes) is not int or num_classes != 10 or type(num_tasks) is not int or num_tasks != 4:
            raise ValueError("AES requires four heads and ten integer classes 0--9")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(learning_rate) not in (int, float) or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if (encoder is None) != (tokenizer is None):
            raise ValueError("Supply both encoder and tokenizer, or neither")
        if encoder is None and (not isinstance(encoder_revision, str) or
                                len(encoder_revision) != 40 or
                                any(c not in "0123456789abcdef" for c in encoder_revision)):
            raise ValueError("Pretrained AES encoder requires an immutable commit revision")
        self.max_length = positive_integer(max_length, "max_length")
        if self.max_length < 3:
            raise ValueError("max_length must accommodate content and special tokens")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder_name, self.encoder_revision = encoder_name, encoder_revision
        if encoder is None:
            encoder = cast(PreTrainedModel, AutoModel.from_pretrained(encoder_name, revision=encoder_revision))
            tokenizer = AutoTokenizer.from_pretrained(encoder_name, revision=encoder_revision)
        self.tokenizer = cast(PreTrainedTokenizerBase, tokenizer)
        position_limit = int(getattr(encoder.config, "max_position_embeddings", self.max_length))
        if encoder.config.model_type == "roberta":
            position_limit -= int(encoder.config.pad_token_id or 0) + 1
        if self.max_length > min(position_limit, self.tokenizer.model_max_length):
            raise ValueError("max_length exceeds the encoder/tokenizer position limit")
        embeddings = encoder.get_input_embeddings()
        if not isinstance(embeddings, nn.Embedding):
            raise ValueError("AES encoder must expose token embeddings")
        if len(self.tokenizer) > embeddings.num_embeddings:
            raise ValueError("Tokenizer vocabulary exceeds encoder embeddings")
        if self.tokenizer.truncation_side != "right":
            raise ValueError("AES requires right truncation")
        self.model = OrdinalRegressionModel(num_classes, num_tasks, encoder=encoder)
        self.model.to(self.device)
        self.num_classes, self.num_tasks = num_classes, num_tasks
        self.learning_rate, self.seed = float(learning_rate), seed
        self.loss_history: list[float] = []
        self.training_metadata: dict[str, Any] = {"status": "untrained"}
        self.bundle_manifest: dict[str, Any] | None = None

    @staticmethod
    def _mapping(mapping, num_classes=10):
        if (num_classes != 10 or mapping != {i: i for i in range(10)} or
                any(type(k) is not int or type(v) is not int for k, v in mapping.items())):
            raise ValueError("AES band mapping must be the identity for integer scores 0--9")

    @staticmethod
    def ordinal_encode(y, band2idx, num_classes):
        OrdinalRegression._mapping(band2idx, num_classes)
        if any(isinstance(v, (bool, np.bool_)) for v in y):
            raise ValueError("Boolean labels are not scores")
        values = np.asarray(y, dtype=float)
        if (values.ndim != 1 or not np.isfinite(values).all() or
                ((values < 0) | (values > 9) | (values != np.floor(values))).any()):
            raise ValueError("AES labels must be integers 0--9; fractional labels are not rounded")
        return (np.arange(9)[None, :] < values[:, None]).astype(np.float32)

    @staticmethod
    def ordinal_encode_multi(y_df, band2idx, num_classes):
        if tuple(y_df.columns) not in (CRITERION_IDS, PUBLIC_CRITERION_IDS):
            raise ValueError("AES criterion order must match the dataset")
        return np.stack([OrdinalRegression.ordinal_encode(y_df[col].tolist(), band2idx, num_classes)
                         for col in y_df.columns], axis=1)

    @staticmethod
    def ordinal_decode(logits, idx2band):
        OrdinalRegression._mapping(idx2band)
        if logits.ndim != 2 or logits.shape[1] != 9 or not torch.isfinite(logits).all():
            raise ValueError("Expected finite ordinal logits with nine thresholds")
        return (torch.sigmoid(logits) > 0.5).sum(dim=1).cpu().tolist()

    @staticmethod
    def ordinal_decode_multi(logits, idx2band):
        OrdinalRegression._mapping(idx2band)
        if logits.ndim != 3 or tuple(logits.shape[1:]) != (4, 9) or not torch.isfinite(logits).all():
            raise ValueError("Expected finite AES logits with shape (N, 4, 9)")
        return (torch.sigmoid(logits) > 0.5).sum(dim=2).cpu().tolist()

    def fit(self, texts, y_ord, epochs=3, batch_size=8):
        validate_texts(texts)
        positive_integer(epochs, "epochs")
        positive_integer(batch_size, "batch_size")
        targets = np.asarray(y_ord, dtype=np.float32)
        if (not texts or targets.shape != (len(texts), 4, 9) or
                not np.isin(targets, [0, 1]).all() or (np.diff(targets, axis=2) > 0).any()):
            raise ValueError("Expected nonempty train and cumulative binary targets (N, 4, 9)")
        lengths = token_lengths(self.tokenizer, texts)
        dataset = EssayDataset(texts, targets, tokenizer=self.tokenizer, max_length=self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=self.learning_rate / 2)
        criterion = nn.BCEWithLogitsLoss()
        self.loss_history = []
        self.training_metadata = {"status": "training_incomplete"}
        self.model.train()
        for epoch in range(epochs):
            total = 0.0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device))
                loss = criterion(logits, batch["labels"].to(self.device))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite AES training loss")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch["input_ids"])
            self.loss_history.append(total / len(dataset))
            scheduler.step()
            print(f"AES epoch {epoch + 1}/{epochs}: loss={self.loss_history[-1]:.6f}", flush=True)
        self.model.eval()
        self.training_metadata = {
            "status": "trained", "train_count": len(texts), "epochs": epochs,
            "batch_size": batch_size, "seed": self.seed, "learning_rate": self.learning_rate,
            "optimizer": "AdamW", "scheduler": "CosineAnnealingLR; per epoch; eta_min=lr/2",
            "loss": "BCEWithLogitsLoss", "loss_history": self.loss_history,
            "lengths": length_summary(lengths, self.max_length),
        }

    def predict_logits(self, texts, batch_size=8):
        validate_texts(texts)
        positive_integer(batch_size, "batch_size")
        dataset = EssayDataset(texts, None, tokenizer=self.tokenizer, max_length=self.max_length)
        self.model.eval()
        outputs = []
        with torch.inference_mode():
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                outputs.append(self.model(batch["input_ids"].to(self.device),
                                          batch["attention_mask"].to(self.device)).cpu())
        return torch.cat(outputs) if outputs else torch.empty((0, 4, 9))

    def predict(self, texts, idx2band=None, batch_size=8) -> list[list[int]]:
        return self.ordinal_decode_multi(self.predict_logits(texts, batch_size),
                                         {i: i for i in range(10)} if idx2band is None else idx2band)

    def predict_essay(self, essay):
        length = token_lengths(self.tokenizer, [essay])[0]
        return {"scores": dict(zip(PUBLIC_CRITERION_IDS, self.predict([essay])[0])),
                "input": {"original_tokens": length, "used_tokens": min(length, self.max_length),
                          "max_length": self.max_length, "truncated": length > self.max_length},
                "semantics": "essay_only; task_response is a legacy proxy, not task-aware TR"}

    def save_bundle(self, save_dir, provenance=None):
        from .inference import save_bundle
        return save_bundle(self, save_dir, provenance=provenance)

    @classmethod
    def load_bundle(cls, save_dir, device=None):
        from .inference import load_bundle
        return load_bundle(save_dir, device=device)

    def save_model_and_tokenizer(self, model_wrapper, save_dir):
        """Compatibility entrypoint; writes the complete versioned bundle."""
        return model_wrapper.save_bundle(save_dir)

    @staticmethod
    def load_model_and_tokenizer(save_dir, num_classes=10, device=None):
        if num_classes != 10:
            raise ValueError("AES supports ten integer classes")
        model = OrdinalRegression.load_bundle(save_dir, device=device)
        return model, model.tokenizer
