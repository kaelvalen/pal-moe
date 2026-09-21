"""
Maximally Interfered Retrieval (MIR) baseline (Aljundi et al., 2019).

Instead of replaying a random buffer batch, MIR performs a virtual gradient
step on the current batch and replays the stored samples whose loss increases
the most under that step (the samples "maximally interfered with" by the
update). This is the strongest simple rehearsal-selection baseline in the
replay family.

Implementation notes:
- the virtual step is a plain SGD step with the trainer learning rate, applied
  in place and then restored (the original formulation);
- the candidate pool is a random subset of the buffer (`candidate_pool`) so the
  selection stays O(pool) per optimisation step;
- the real step optimises CE(current) + CE(selected replay), matching the ER
  baseline's loss so the comparison isolates the selection policy.

Interface matches :class:`pal_moe.baselines.replay.ReplayTrainer`.
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffer import SampleBuffer


class MIR:
    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 200,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        sampling: str = "recency",
        candidate_pool: int = 128,
    ):
        self.model = model
        self.buffer_size = buffer_size
        self.lr = lr
        self.device = device
        self.candidate_pool = candidate_pool
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.buffer = SampleBuffer(buffer_size, mode=sampling)

    def update_buffer(self, train_loader: Any) -> None:
        """Stores `buffer_size // 5` random exemplars from the current task."""
        collected_x, collected_y = [], []
        for x, y in train_loader:
            collected_x.append(x)
            collected_y.append(y)
        if not collected_x:
            return
        self.buffer.add_task(
            torch.cat(collected_x, dim=0),
            torch.cat(collected_y, dim=0),
            per_task_budget=max(1, self.buffer_size // 5),
        )

    def _select_interfered(
        self, batch_size: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Virtual-step loss increase ranking over a candidate pool."""
        cx, cy, _ = self.buffer.sample_tensors(self.candidate_pool, self.device)
        if cx is None or cy is None:
            return None, None
        params = [p for p in self.model.parameters() if p.requires_grad]
        backup = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.data -= self.lr * p.grad
        with torch.no_grad():
            candidate_loss = F.cross_entropy(self.model(cx), cy, reduction="none")
        with torch.no_grad():
            for p, b in zip(params, backup):
                p.data.copy_(b)
        k = min(batch_size, candidate_loss.numel())
        if k <= 0:
            return None, None
        idx = candidate_loss.topk(k).indices
        return cx[idx], cy[idx]

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

                # Gradient on the current batch, used for the virtual step.
                self.optimizer.zero_grad()
                loss = F.cross_entropy(self.model(x), y)
                loss.backward()

                rx, ry = self._select_interfered(batch_size=x.size(0) // 2)

                # Real update: CE(current) + CE(selected replay).
                self.optimizer.zero_grad()
                loss = F.cross_entropy(self.model(x), y)
                if rx is not None and ry is not None:
                    loss = loss + F.cross_entropy(self.model(rx), ry)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach()
                n_updates += 1

        self.update_buffer(train_loader)
        return {
            "task_id": task_id,
            "loss": float(total_loss.item()) / max(n_updates, 1),
        }

    def memory_bytes(self) -> int:
        """Stored exemplar bytes (selection is computed on the fly)."""
        return self.buffer.memory_bytes()
