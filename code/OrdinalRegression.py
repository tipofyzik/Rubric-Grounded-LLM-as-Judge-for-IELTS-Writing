import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os



# model_name = "roberta-base-uncased"
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# =====================
# DATASET
# =====================
class EssayDataset(Dataset):
    def __init__(self, texts: list[str], targets: np.ndarray) -> None:
        """
        Initializes the EssayDataset instance.

        Arguments:
            texts (list[str]): A list of input essay texts.
            targets (np.ndarray): Ordinal encoded targets. The expected shape is
                (num_samples, num_tasks, num_classes - 1) or
                (num_samples, num_classes - 1) depending on the task setup.
        """
        self.texts = texts
        self.targets = targets
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Total number of text samples.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retrieves a tokenized sample and its corresponding target.

        Arguments:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - input_ids (torch.Tensor): Token IDs tensor of shape (seq_len).
                - attention_mask (torch.Tensor): Attention mask tensor of shape (seq_len).
                - labels (torch.Tensor): Target tensor.
        """
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(
            self.targets[idx],
            dtype=torch.float
        )
        return item

# ========================
# ORDINAL REGRESSION MODEL
# ========================
class OrdinalHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes - 1)  # for ordinal regression
        )

    def forward(self, x):
        return self.net(x)
        
class OrdinalRegressionModel(nn.Module):
    """
    """
    def __init__(self, num_classes: int, num_tasks: int) -> None:
        """
        Initializes the OrdinalRegressionModel.

        Arguments:
            num_classes (int): Total number of ordinal classes.
            num_tasks (int): Number of prediction tasks.
        """
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.num_tasks = num_tasks
        self.num_classes = num_classes


        self.heads = nn.ModuleList([
            OrdinalHead(hidden_dim=hidden, num_classes=num_classes)
            for _ in range(num_tasks)
        ])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Arguments:
            input_ids (torch.Tensor): Token ID tensor of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_tasks, num_classes - 1).
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        hidden = hidden * mask
        cls = hidden.sum(dim=1) / mask.sum(dim=1)
        logits = [head(cls) for head in self.heads]
        # (batch, tasks, K-1)
        logits = torch.stack(logits, dim=1)
    
        return logits

# ==========================
# ORDINAL REGRESSION WRAPPER
# ==========================
class OrdinalRegression:
    """
    """
    def __init__(self, num_classes: int, num_tasks: int, device: torch.device | None = None) -> None:
        """
        Initializes the OrdinalRegression wrapper.

        Arguments:
            num_classes (int): Total number of ordinal classes.
            num_tasks (int): Number of prediction tasks.
            device (torch.device | None): Device used for training and inference.
                If None, the device is automatically selected (CUDA if available).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = OrdinalRegressionModel(num_classes, num_tasks).to(self.device)
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-5)
    
    # ===================
    # ENCODING & DECODING
    # ===================
    @staticmethod
    def ordinal_encode(y: list[float], band2idx: dict[float, int], num_classes: int) -> np.ndarray:
        """
        Encodes ordinal labels using cumulative ordinal encoding.

        Arguments:
            y (list[float]): List of ordinal labels (e.g. [0.0, 0.5, 1.0, ...]).
            band2idx (dict[float, int]): Dictionary mapping label values to ordinal indices.
            num_classes (int): Total number of ordinal classes.

        Returns:
            np.ndarray: Encoded array with shape (num_samples, num_classes - 1).
        """
        encoded = np.zeros((len(y), num_classes - 1), dtype=np.float32)
        for i, label in enumerate(y):
            idx = band2idx[label]
            encoded[i, :idx] = 1.0
        return encoded

    @staticmethod
    def ordinal_encode_multi(y_df, band2idx: dict, num_classes: int) -> np.ndarray:
        """
        Encodes multiple ordinal targets using cumulative encoding.

        Arguments:
            y_df (pandas.DataFrame): DataFrame containing ordinal targets where
                each column corresponds to a prediction task.
            band2idx (dict): Dictionary mapping label values to ordinal indices.
            num_classes (int): Total number of ordinal classes.

        Returns:
            np.ndarray: Encoded tensor with shape
                (num_samples, num_tasks, num_classes - 1).
        """
        num_targets = y_df.shape[1]
    
        encoded = np.zeros(
            (len(y_df), num_targets, num_classes - 1),
            dtype=np.float32
        )
    
        for t, col in enumerate(y_df.columns):
            for i, label in enumerate(y_df[col]):
                idx = band2idx[label]
                encoded[i, t, :idx] = 1.0
    
        return encoded
    
    @staticmethod
    def ordinal_decode(logits: torch.Tensor, idx2band: dict) -> list:
        """
        Decodes ordinal logits into predicted labels.

        Arguments:
            logits (torch.Tensor): Logits tensor of shape (batch_size, num_classes - 1).
            idx2band (dict): Dictionary mapping ordinal indices to original label values.

        Returns:
            list: List of predicted labels.
        """
        probs = torch.sigmoid(logits)
        preds_idx = (probs > 0.5).sum(dim=1).cpu().tolist()
        return [idx2band[int(i)] for i in preds_idx]

    @staticmethod
    def ordinal_decode_multi(logits: torch.Tensor, idx2band: dict) -> list[list[float]]:
        """
        Decodes multi-task ordinal logits into predicted labels.

        Arguments:
            logits (torch.Tensor): Logits tensor of shape
                (batch_size, num_tasks, num_classes - 1).
            idx2band (dict): Dictionary mapping ordinal indices to original label values.

        Returns:
            list[list[float]]: Predicted labels for each sample and task.
                Shape: (num_samples, num_tasks).
        """
        probs = torch.sigmoid(logits)
    
        preds_idx = (probs > 0.5).sum(dim=2).cpu().numpy()
    
        results = []
        for sample in preds_idx:
            results.append([
                float(idx2band[int(i)])   # ⭐ ключевая строка
                for i in sample
            ])
    
        return results
    
    # ========
    # TRAINING
    # ========
    def fit(self, texts: list[str], y_ord: np.ndarray, 
            epochs: int = 3, batch_size: int = 8) -> None:
        """
        Trains the ordinal regression model.

        Arguments:
            texts (list[str]): List of input training texts.
            y_ord (np.ndarray): Ordinal encoded targets with shape
                (num_samples, num_tasks, num_classes - 1).
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.

        Returns:
            None
        """
        dataset = EssayDataset(texts, y_ord)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        
        self.loss_history = []
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            epoch_loss = total_loss / len(loader)
            self.loss_history.append(epoch_loss) 
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    # ==========
    # PREDICTION
    # ==========
    def predict(self, texts: list[str], idx2band: dict, 
                batch_size: int = 8) -> list[list[float]]:
        """
        Predicts ordinal labels for input texts.

        Arguments:
            texts (list[str]): List of input texts for prediction.
            idx2band (dict): Dictionary mapping ordinal indices to original labels.
            batch_size (int): Batch size used during inference.

        Returns:
            list[list[float]]: Predicted labels for each sample and task.
                Shape: (num_samples, num_tasks).
        """
        dataset = EssayDataset(texts, np.zeros((len(texts), self.num_classes - 1), dtype=np.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(input_ids, attention_mask)
                batch_preds = self.ordinal_decode_multi(logits, idx2band)
                preds.extend(batch_preds)
        return preds
    
    # ==============
    # SAVING RESULTS
    # ==============
    def save_model_and_tokenizer(self, model_wrapper: "OrdinalRegression", save_dir: str) -> None:
        """
        Saves the trained model and tokenizer to disk.

        Arguments:
            model_wrapper (OrdinalRegression): Trained model wrapper instance.
            save_dir (str): Directory where the model and tokenizer will be saved.

        Returns:
            None
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Saving model weights
        torch.save(model_wrapper.model.state_dict(), os.path.join(save_dir, "ordinal_model.pt"))
        
        # Saving tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_dir)
        
        print(f"Model and tokenizer saved to {save_dir}")

    def load_model_and_tokenizer(self, save_dir, num_classes, device=None):
        """
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_wrapper = OrdinalRegression(num_classes=num_classes, num_tasks=4, device=device)
        
        # Loading weights
        state_dict = torch.load(os.path.join(save_dir, "ordinal_model.pt"), map_location=device)
        model_wrapper.model.load_state_dict(state_dict)
        
        # Loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        
        return model_wrapper, tokenizer

