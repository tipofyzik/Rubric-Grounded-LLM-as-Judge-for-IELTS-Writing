"""Essay-only AES: four ordinal heads and integer scores in dataset order."""

import random
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ielts_judge.Criteria import CRITERION_IDS, PUBLIC_CRITERION_IDS

ENCODER_ID = "FacebookAI/roberta-base"
ENCODER_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"


def positive_integer(value: Any, name: Any) -> Any:
    """Validate and return a strictly positive integer without coercion.

    Args:
        value (Any): Value used by this operation.
        name (Any): Name used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_texts(texts: Any) -> None:
    """Require a nonempty English essay string for each requested model input.

    Args:
        texts (Any): Texts used by this operation.

    Returns:
        None: This operation returns no value.
    """
    if not isinstance(texts, list) or any(not isinstance(essay_text, str) or not essay_text.strip() for essay_text in texts):
        raise ValueError("AES expects a list of nonempty essay strings")


def token_lengths(tokenizer: Any, texts: Any, batch_size: int=128) -> Any:
    """Count complete inputs, including special tokens, without truncation.

    Args:
        tokenizer (Any): Tokenizer used by this operation.
        texts (Any): Texts used by this operation.
        batch_size (int): Batch size used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    validate_texts(texts)
    positive_integer(batch_size, "batch_size")
    lengths = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], truncation=False,
                            padding=False, verbose=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def length_summary(lengths: Any, max_length: int) -> dict[str, Any]:
    """Summarize original token lengths and count inputs exceeding the model limit.

    Args:
        lengths (Any): Lengths used by this operation.
        max_length (int): Max length used by this operation.

    Returns:
        dict[str, Any]: Named result fields and metadata produced by this operation.
    """
    return {"count": len(lengths), "truncated_count": sum(token_length > max_length for token_length in lengths),
            "max_original_tokens": max(lengths, default=0), "max_length": max_length}


class EssayDataset(Dataset):
    """Tokenize essay examples and attach the four ordinal training targets."""
    def __init__(self, texts: Any, targets: Any, *, tokenizer: Any, max_length: int=512) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            texts (Any): Texts used by this operation.
            targets (Any): Targets used by this operation.
            tokenizer (Any): Tokenizer used by this operation.
            max_length (int): Max length used by this operation.

        Returns:
            None: This operation returns no value.
        """
        validate_texts(texts)
        self.texts, self.targets = texts, targets
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self) -> int:
        """Return the number of examples available to the model data loader.

        Returns:
            int: The computed result for the supplied inputs.
        """
        return len(self.texts)

    def __getitem__(self, current_index: Any) -> Any:
        """Tokenize one indexed example and return its model inputs and labels.

        Args:
            current_index (Any): Current index used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        encoded = self.tokenizer(self.texts[current_index], truncation=True,
                                 padding="max_length", max_length=self.max_length,
                                 return_tensors="pt")
        item = {key: encoded[key].squeeze(0) for key in ("input_ids", "attention_mask")}
        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[current_index], dtype=torch.float32)
        return item


class OrdinalHead(nn.Module):
    """Manage ordinal head through the operations defined by this component."""
    def __init__(self, hidden_dim: Any, num_classes: Any, dropout: float=0.2) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            hidden_dim (Any): Hidden dim used by this operation.
            num_classes (Any): Num classes used by this operation.
            dropout (float): Dropout used by this operation.

        Returns:
            None: This operation returns no value.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes - 1),
        )

    def forward(self, sample: Any) -> Any:
        """Compute model outputs for the supplied input tensors.

        Args:
            sample (Any): Sample used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        return self.net(sample)


class OrdinalRegressionModel(nn.Module):
    """Manage ordinal regression model through the operations defined by this component."""
    def __init__(self, num_classes: int, num_tasks: int, *, encoder: PreTrainedModel) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            num_classes (int): Num classes used by this operation.
            num_tasks (int): Num tasks used by this operation.
            encoder (PreTrainedModel): Encoder used by this operation.

        Returns:
            None: This operation returns no value.
        """
        super().__init__()
        self.encoder = encoder
        self.num_tasks, self.num_classes = num_tasks, num_classes
        hidden = encoder.config.hidden_size
        self.heads = nn.ModuleList([OrdinalHead(hidden, num_classes) for _ in range(num_tasks)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute model outputs for the supplied input tensors.

        Args:
            input_ids (torch.Tensor): Input ids used by this operation.
            attention_mask (torch.Tensor): Attention mask used by this operation.

        Returns:
            torch.Tensor: The computed result for the supplied inputs.
        """
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return torch.stack([head(pooled) for head in self.heads], dim=1)


class OrdinalRegression:
    """Manage ordinal regression through the operations defined by this component."""
    def __init__(self, num_classes: int=10, num_tasks: int=4, device: Any=None, *,
                 encoder_name: Any=ENCODER_ID, encoder_revision: Any=ENCODER_REVISION,
                 encoder: PreTrainedModel | None = None,
                 tokenizer: PreTrainedTokenizerBase | None = None,
                 max_length: int=512, learning_rate: float=2e-5, seed: int=42) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            num_classes (int): Num classes used by this operation.
            num_tasks (int): Num tasks used by this operation.
            device (Any): Device used by this operation.
            encoder_name (Any): Encoder name used by this operation.
            encoder_revision (Any): Encoder revision used by this operation.
            encoder (PreTrainedModel | None): Encoder used by this operation.
            tokenizer (PreTrainedTokenizerBase | None): Tokenizer used by this operation.
            max_length (int): Max length used by this operation.
            learning_rate (float): Learning rate used by this operation.
            seed (int): Seed used by this operation.

        Returns:
            None: This operation returns no value.
        """
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
                                any(current_criterion not in "0123456789abcdef" for current_criterion in encoder_revision)):
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
    def _mapping(mapping: Any, num_classes: int=10) -> None:
        """Mapping.

        Args:
            mapping (Any): Mapping used by this operation.
            num_classes (int): Num classes used by this operation.

        Returns:
            None: This operation returns no value.
        """
        if (num_classes != 10 or mapping != {current_index: current_index for current_index in range(10)} or
                any(type(current_key) is not int or type(current_value) is not int for current_key, current_value in mapping.items())):
            raise ValueError("AES band mapping must be the identity for integer scores 0--9")

    @staticmethod
    def ordinal_encode(current_target: Any, band2idx: Any, num_classes: Any) -> Any:
        """Ordinal encode.

        Args:
            current_target (Any): Current target used by this operation.
            band2idx (Any): Band2idx used by this operation.
            num_classes (Any): Num classes used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        OrdinalRegression._mapping(band2idx, num_classes)
        if any(isinstance(current_value, (bool, np.bool_)) for current_value in current_target):
            raise ValueError("Boolean labels are not scores")
        values = np.asarray(current_target, dtype=float)
        if (values.ndim != 1 or not np.isfinite(values).all() or
                ((values < 0) | (values > 9) | (values != np.floor(values))).any()):
            raise ValueError("AES labels must be integers 0--9; fractional labels are not rounded")
        return (np.arange(9)[None, :] < values[:, None]).astype(np.float32)

    @staticmethod
    def ordinal_encode_multi(y_df: Any, band2idx: Any, num_classes: Any) -> Any:
        """Ordinal encode multi.

        Args:
            y_df (Any): Y df used by this operation.
            band2idx (Any): Band2idx used by this operation.
            num_classes (Any): Num classes used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        if tuple(y_df.columns) not in (CRITERION_IDS, PUBLIC_CRITERION_IDS):
            raise ValueError("AES criterion order must match the dataset")
        return np.stack([OrdinalRegression.ordinal_encode(y_df[column].tolist(), band2idx, num_classes)
                         for column in y_df.columns], axis=1)

    @staticmethod
    def ordinal_decode(logits: Any, idx2band: Any) -> Any:
        """Ordinal decode.

        Args:
            logits (Any): Logits used by this operation.
            idx2band (Any): Idx2band used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        OrdinalRegression._mapping(idx2band)
        if logits.ndim != 2 or logits.shape[1] != 9 or not torch.isfinite(logits).all():
            raise ValueError("Expected finite ordinal logits with nine thresholds")
        return (torch.sigmoid(logits) > 0.5).sum(dim=1).cpu().tolist()

    @staticmethod
    def ordinal_decode_multi(logits: Any, idx2band: Any) -> Any:
        """Ordinal decode multi.

        Args:
            logits (Any): Logits used by this operation.
            idx2band (Any): Idx2band used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        OrdinalRegression._mapping(idx2band)
        if logits.ndim != 3 or tuple(logits.shape[1:]) != (4, 9) or not torch.isfinite(logits).all():
            raise ValueError("Expected finite AES logits with shape (N, 4, 9)")
        return (torch.sigmoid(logits) > 0.5).sum(dim=2).cpu().tolist()

    def fit(self, texts: Any, y_ord: Any, epochs: int=3, batch_size: int=8) -> None:
        """Fit.

        Args:
            texts (Any): Texts used by this operation.
            y_ord (Any): Y ord used by this operation.
            epochs (int): Epochs used by this operation.
            batch_size (int): Batch size used by this operation.

        Returns:
            None: This operation returns no value.
        """
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

    def predict_logits(self, texts: Any, batch_size: int=8) -> Any:
        """Predict logits.

        Args:
            texts (Any): Texts used by this operation.
            batch_size (int): Batch size used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
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

    def predict(self, texts: Any, idx2band: Any=None, batch_size: int=8) -> list[list[int]]:
        """Predict.

        Args:
            texts (Any): Texts used by this operation.
            idx2band (Any): Idx2band used by this operation.
            batch_size (int): Batch size used by this operation.

        Returns:
            list[list[int]]: The computed result for the supplied inputs.
        """
        return self.ordinal_decode_multi(self.predict_logits(texts, batch_size),
                                         {current_index: current_index for current_index in range(10)} if idx2band is None else idx2band)

    def predict_essay(self, essay: str) -> dict[str, Any]:
        """Return essay-only AES estimates with the model's input-length metadata.

        Args:
            essay (str): English essay text supplied for evaluation.

        Returns:
            dict[str, Any]: Named result fields and metadata produced by this operation.
        """
        length = token_lengths(self.tokenizer, [essay])[0]
        return {"scores": dict(zip(PUBLIC_CRITERION_IDS, self.predict([essay])[0])),
                "input": {"original_tokens": length, "used_tokens": min(length, self.max_length),
                          "max_length": self.max_length, "truncated": length > self.max_length},
                "semantics": "essay_only; task_response is a legacy proxy, not task-aware TR"}

    def save_bundle(self, save_dir: Any, provenance: Any=None) -> Any:
        """Save bundle.

        Args:
            save_dir (Any): Save dir used by this operation.
            provenance (Any): Provenance used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from ielts_judge.aes.Inference import save_bundle
        return save_bundle(self, save_dir, provenance=provenance)

    @classmethod
    def load_bundle(cls, save_dir: Any, device: Any=None) -> Any:
        """Load bundle.

        Args:
            save_dir (Any): Save dir used by this operation.
            device (Any): Device used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        from ielts_judge.aes.Inference import load_bundle
        return load_bundle(save_dir, device=device)

    def save_model_and_tokenizer(self, model_wrapper: Any, save_dir: Any) -> Any:
        """Compatibility entrypoint; writes the complete versioned bundle.

        Args:
            model_wrapper (Any): Model wrapper used by this operation.
            save_dir (Any): Save dir used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        return model_wrapper.save_bundle(save_dir)

    @staticmethod
    def load_model_and_tokenizer(save_dir: Any, num_classes: int=10, device: Any=None) -> Any:
        """Load model and tokenizer.

        Args:
            save_dir (Any): Save dir used by this operation.
            num_classes (int): Num classes used by this operation.
            device (Any): Device used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        if num_classes != 10:
            raise ValueError("AES supports ten integer classes")
        model = OrdinalRegression.load_bundle(save_dir, device=device)
        return model, model.tokenizer
