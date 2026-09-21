"""
Average Gradient Episodic Memory (A-GEM) baseline (Chaudhry et al., 2019).

Projects the gradient of the current batch onto the half-space defined by a
reference batch sampled from episodic memory: the update is constrained so it
never increases the loss on remembered samples.

Interface matches pal_moe.baselines.replay.ReplayTrainer:
    trainer.train_task(task_id, train_loader, epochs)
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffer import SampleBuffer


class AGEM:
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

    def update_buffer(self, train_loader: Any, seen_tasks: int) -> None:
        collected_x, collected_y = [], []
        for x, y in train_loader:
            collected_x.append(x)
            collected_y.append(y)
        cat_x = torch.cat(collected_x, dim=0)
        cat_y = torch.cat(collected_y, dim=0)

        self.buffer.add_task(cat_x, cat_y, seen_tasks=seen_tasks)

    def get_ref_batch(
        self, batch_size: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        rx, ry, _ = self.buffer.sample_tensors(batch_size, self.device)
        return rx, ry

    def memory_bytes(self) -> int:
        """Stored exemplar bytes."""
        return self.buffer.memory_bytes()

    def _flatten_grad(self) -> torch.Tensor:
        return torch.cat(
            [p.grad.flatten() for p in self.model.parameters() if p.grad is not None]
        )

    def _unflatten_grad(self, gvec: torch.Tensor) -> None:
        idx = 0
        for p in self.model.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.data.copy_(gvec[idx : idx + n].view_as(p))
                idx += n

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

                # 1. Gradient on the current batch
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                grad_new = self._flatten_grad()

                # 2. Reference gradient from episodic memory
                rx, ry = self.get_ref_batch(batch_size=64)
                if rx is not None:
                    self.optimizer.zero_grad()
                    ref_logits = self.model(rx)
                    loss_ref = F.cross_entropy(ref_logits, ry)
                    loss_ref.backward()
                    grad_ref = self._flatten_grad()
                    if grad_ref.norm() > 1e-12:
                        dot = (grad_new * grad_ref).sum()
                        if dot < 0:
                            g_ref_norm_sq = (grad_ref * grad_ref).sum().clamp(min=1e-12)
                            projected = grad_new - (dot / g_ref_norm_sq) * grad_ref
                            self._unflatten_grad(projected)

                self.optimizer.step()
                total_loss += loss.detach()
                n_updates += 1

        self.update_buffer(train_loader, seen_tasks=task_id + 1)
        return {
            "task_id": task_id,
            "loss": float(total_loss.item()) / max(n_updates, 1),
        }
