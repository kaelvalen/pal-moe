"""
Naive Sequential Fine-tuning baseline.
Trains sequentially across tasks without any forgetting mitigation mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict


class NaiveFineTuning:
    """
    Standard sequential fine-tuning baseline.
    Serves as empirical lower bound exhibiting classic catastrophic forgetting.
    """
    def __init__(self, model: nn.Module, lr: float = 1e-3, device: torch.device = torch.device("cpu")):
        self.model = model
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_task(self, task_id: int, train_loader: Any, epochs: int = 5) -> Dict[str, Any]:
        self.model.train()
        losses = []
        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

        return {"task_id": task_id, "loss": sum(losses) / max(len(losses), 1)}
