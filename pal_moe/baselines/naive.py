"""
Naive Sequential Fine-tuning baseline.
Trains sequentially across tasks without any forgetting mitigation mechanism.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveFineTuning:
    """
    Standard sequential fine-tuning baseline.
    Serves as empirical lower bound exhibiting classic catastrophic forgetting.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_task(
        self, task_id: int, train_loader: Any, epochs: int = 5
    ) -> dict[str, Any]:
        self.model.train()
        # Accumulate on-device and convert once: a per-batch loss.item() forces
        # a host synchronisation on every optimisation step.
        total_loss = torch.zeros((), device=self.device)
        n_updates = 0
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True
                )
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach()
                n_updates += 1

        return {
            "task_id": task_id,
            "loss": float(total_loss.item()) / max(n_updates, 1),
        }
