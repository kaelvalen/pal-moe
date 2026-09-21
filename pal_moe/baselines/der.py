"""
Dark Experience Replay++ (DER++) and Asymmetric Cross-Entropy Replay (ER-ACE).

Modern memory-based continual learning baselines:
- DER (Buzzega et al., 2020): replay of past logits via MSE distillation.
- DER++: DER + direct labels (CE) on buffered samples.
- ER-ACE (Arslan et al., 2022): asymmetric cross-entropy; buffered old samples are
  trained to have low probability on the CURRENT task's classes ("gained" classes),
  preventing interference with past knowledge.

Same interface as pal_moe.baselines.replay.ReplayTrainer:
    trainer.train_task(task_id, train_loader, epochs)
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffer import SampleBuffer


class DERPP:
    """Dark Experience Replay++ (DER + labels)."""

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 200,
        lr: float = 1e-3,
        alpha: float = 0.5,
        beta: float = 0.5,
        device: torch.device = torch.device("cpu"),
        sampling: str = "recency",
    ):
        self.model = model
        self.buffer_size = buffer_size
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.buffer = SampleBuffer(buffer_size, mode=sampling)

    def update_buffer(
        self, train_loader: Any, seen_tasks: int, logit_batch_size: int = 512
    ) -> None:
        """
        Stores random exemplars from current task with the CURRENT model's logits.

        The logits are computed in fixed-size chunks: a single forward over the
        whole task split (e.g. ~9000 CIFAR images) needs multi-GB activations
        and can OOM even with the graph disabled.
        """
        collected_x, collected_y = [], []
        for x, y in train_loader:
            collected_x.append(x)
            collected_y.append(y)
        cat_x = torch.cat(collected_x, dim=0)
        cat_y = torch.cat(collected_y, dim=0)

        logit_chunks = []
        with torch.no_grad():
            for start in range(0, cat_x.size(0), logit_batch_size):
                chunk = cat_x[start : start + logit_batch_size].to(self.device)
                logit_chunks.append(self.model(chunk).detach().cpu())
        logits_all = torch.cat(logit_chunks, dim=0)

        self.buffer.add_task(cat_x, cat_y, logits_all, seen_tasks=seen_tasks)

    def get_replay_batch(
        self, batch_size: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.buffer.sample_tensors(batch_size, self.device)

    def memory_bytes(self) -> int:
        """Stored exemplars plus the cached logits (DER++ keeps both)."""
        return self.buffer.memory_bytes()

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
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)

                rx, ry, rl = self.get_replay_batch(batch_size=x.size(0) // 2)
                if rx is not None:
                    r_logits = self.model(rx)
                    loss = loss + self.alpha * F.mse_loss(r_logits, rl)
                    if ry is not None:
                        loss = loss + self.beta * F.cross_entropy(r_logits, ry)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach()
                n_updates += 1

        self.update_buffer(train_loader, seen_tasks=task_id + 1)
        return {
            "task_id": task_id,
            "loss": float(total_loss.item()) / max(n_updates, 1),
        }


class ERACE:
    """Experience Replay with Asymmetric Cross-Entropy (ER-ACE)."""

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
        self.classes_per_task: list[list[int]] = []

    def update_buffer(
        self, train_loader: Any, seen_tasks: int, current_classes: list[int]
    ) -> None:
        collected_x, collected_y = [], []
        for x, y in train_loader:
            collected_x.append(x)
            collected_y.append(y)
        cat_x = torch.cat(collected_x, dim=0)
        cat_y = torch.cat(collected_y, dim=0)

        self.buffer.add_task(cat_x, cat_y, seen_tasks=seen_tasks)
        self.classes_per_task.append(list(current_classes))

    def get_replay_batch(
        self, batch_size: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bx, by, _ = self.buffer.sample_tensors(batch_size, self.device)
        return bx, by

    def memory_bytes(self) -> int:
        """Stored exemplar bytes."""
        return self.buffer.memory_bytes()

    def train_task(
        self,
        task_id: int,
        train_loader: Any,
        epochs: int = 5,
        current_classes: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        if current_classes is None:
            # heuristic default: 10-class problems split into +2 classes per task
            current_classes = [2 * task_id, 2 * task_id + 1]
        self.model.train()
        self.classes_per_task.append(list(current_classes))
        total_loss = torch.zeros((), device=self.device)
        n_updates = 0
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True
                )
                self.optimizer.zero_grad()
                logits = self.model(x)
                # Standard CE over all classes for current-task samples
                loss = F.cross_entropy(logits, y)

                # Asymmetric cross-entropy for buffered (old) samples:
                # they must have LOW probability mass on the current task's classes.
                rx, ry = self.get_replay_batch(batch_size=x.size(0) // 2)
                if rx is not None:
                    r_logits = self.model(rx)
                    p = F.softmax(r_logits, dim=-1)
                    p_current = p[:, current_classes].sum(dim=-1)  # [B]
                    loss_buf = -torch.log(1.0 - p_current + 1e-9).mean()
                    loss = loss + loss_buf

                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach()
                n_updates += 1

        self.update_buffer(
            train_loader, seen_tasks=task_id + 1, current_classes=current_classes
        )
        return {
            "task_id": task_id,
            "loss": float(total_loss.item()) / max(n_updates, 1),
        }
