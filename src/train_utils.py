from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


class LogisticClass(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hist = {"loss": []}

    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        batch_size: int = 64,
        epochs: int = 5,
    ) -> Dict[str, list]:
        X_tr_np, y_tr_np = train_data
        X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
        y_tr = torch.tensor(y_tr_np, dtype=torch.float32)

        if y_tr.ndim == 1:
            y_tr = y_tr.view(-1, 1)

        train_loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
        )

        for _ in tqdm(range(epochs), desc="Training", leave=False):
            self.model.train()
            total_loss = 0.0
            total_n = 0
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x).squeeze()
                loss = self.loss_fn(logits, y.squeeze())
                actual_bs = x.size(0)
                total_loss += loss.item() * actual_bs
                total_n += actual_bs
                loss.backward()
                self.optimizer.step()
            self.hist["loss"].append(total_loss / max(total_n, 1))
        return self.hist


def to_dense(array_like):
    if hasattr(array_like, "toarray"):
        return array_like.toarray()
    return array_like
