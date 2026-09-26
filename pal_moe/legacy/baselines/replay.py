"""
Experience Replay baseline.
Maintains a small buffer of raw input exemplars from previous tasks and replays them during training.
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffer import SampleBuffer


class ReplayTrainer:
    """
    Standard rehearsal-based continual learning baseline.

    The per-task budget is `buffer_size // 5`, which is balanced for the
    5-task benchmarks; `sampling="reservoir"` switches to uniform reservoir
    sampling over the whole stream.
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 200,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        sampling: str = "recency",
    ):
        self.model = model
        self.buffer_size = buffer_size
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.buffer = SampleBuffer(buffer_size, mode=sampling)

    def update_buffer(self, train_loader: Any) -> None:
        """Stores `buffer_size // 5` random exemplars from the current task."""
        collected_x = []
        collected_y = []
        for x, y in train_loader:
            collected_x.append(x)
            collected_y.append(y)
        cat_x = torch.cat(collected_x, dim=0)
        cat_y = torch.cat(collected_y, dim=0)

        self.buffer.add_task(
            cat_x,
            cat_y,
            per_task_budget=max(1, self.buffer_size // 5),
        )

    def get_replay_batch(
        self, batch_size: int = 32
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bx, by, _ = self.buffer.sample_tensors(batch_size, self.device)
        return bx, by

    def memory_bytes(self) -> int:
        """Stored exemplar bytes (the only persistent state this baseline keeps)."""
        return self.buffer.memory_bytes()

    def train_task(
        self, task_id: int, train_loader: Any, epochs: int = 5
    ) -> dict[str, Any]:
        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        n_updates = 0
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = (
                    x.to(self.device, non_blocking=True),
                    y.to(self.device, non_blocking=True),
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
