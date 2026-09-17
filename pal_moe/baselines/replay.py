"""
Experience Replay baseline.
Maintains a small buffer of raw input exemplars from previous tasks and replays them during training.
"""

import random
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayTrainer:
    """
    Standard rehearsal-based continual learning baseline.
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 200,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.buffer_size = buffer_size
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.buffer_x: list[torch.Tensor] = []
        self.buffer_y: list[torch.Tensor] = []

    def update_buffer(
        self, train_loader: Any, per_task_budget: Optional[int] = None
    ) -> None:
        """Stores random exemplars from current task."""
        collected_x = []
        collected_y = []
        for x, y in train_loader:
            collected_x.append(x)
            collected_y.append(y)
        cat_x = torch.cat(collected_x, dim=0)
        cat_y = torch.cat(collected_y, dim=0)

        indices = list(range(cat_x.size(0)))
        random.shuffle(indices)
        budget = (
            per_task_budget
            if per_task_budget is not None
            else max(1, self.buffer_size // 5)
        )
        selected = indices[:budget]

        for idx in selected:
            self.buffer_x.append(cat_x[idx].clone())
            self.buffer_y.append(cat_y[idx].clone())

        # Enforce max buffer size
        if len(self.buffer_x) > self.buffer_size:
            self.buffer_x = self.buffer_x[-self.buffer_size :]
            self.buffer_y = self.buffer_y[-self.buffer_size :]

    def get_replay_batch(
        self, batch_size: int = 32
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.buffer_x:
            return None, None
        indices = [
            random.randint(0, len(self.buffer_x) - 1)
            for _ in range(min(batch_size, len(self.buffer_x)))
        ]
        bx = torch.stack([self.buffer_x[i] for i in indices]).to(self.device)
        by = torch.stack([self.buffer_y[i] for i in indices]).to(self.device)
        return bx, by

    def train_task(
        self, task_id: int, train_loader: Any, epochs: int = 5
    ) -> dict[str, Any]:
        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        n_updates = 0
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True
                )
                self.optimizer.zero_grad()

                # Current task forward
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)

                # Replay batch forward
                rx, ry = self.get_replay_batch(batch_size=x.size(0) // 2)
                if rx is not None and ry is not None:
                    r_logits = self.model(rx)
                    loss_replay = F.cross_entropy(r_logits, ry)
                    loss = 0.5 * loss + 0.5 * loss_replay

                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach()
                n_updates += 1

        self.update_buffer(train_loader)
        return {
            "task_id": task_id,
            "loss": float(total_loss.item()) / max(n_updates, 1),
        }
