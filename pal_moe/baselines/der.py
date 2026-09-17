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

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional


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
    ):
        self.model = model
        self.buffer_size = buffer_size
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.buffer_x: List[torch.Tensor] = []
        self.buffer_y: List[torch.Tensor] = []
        self.buffer_logits: List[torch.Tensor] = []

    def update_buffer(self, train_loader: Any, seen_tasks: int) -> None:
        """Stores random exemplars from current task with the CURRENT model's logits."""
        collected_x, collected_y = [], []
        for x, y in train_loader:
            collected_x.append(x)
            collected_y.append(y)
        cat_x = torch.cat(collected_x, dim=0)
        cat_y = torch.cat(collected_y, dim=0)

        budget = max(1, self.buffer_size // max(seen_tasks, 1))
        indices = list(range(cat_x.size(0)))
        random.shuffle(indices)
        selected = indices[:budget]

        with torch.no_grad():
            logits_all = self.model(cat_x.to(self.device)).detach().cpu()

        for idx in selected:
            self.buffer_x.append(cat_x[idx].clone())
            self.buffer_y.append(cat_y[idx].clone())
            self.buffer_logits.append(logits_all[idx].clone())

        if len(self.buffer_x) > self.buffer_size:
            self.buffer_x = self.buffer_x[-self.buffer_size :]
            self.buffer_y = self.buffer_y[-self.buffer_size :]
            self.buffer_logits = self.buffer_logits[-self.buffer_size :]

    def get_replay_batch(
        self, batch_size: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.buffer_x:
            return None, None, None
        n = min(batch_size, len(self.buffer_x))
        indices = [random.randint(0, len(self.buffer_x) - 1) for _ in range(n)]
        bx = torch.stack([self.buffer_x[i] for i in indices]).to(self.device)
        by = torch.stack([self.buffer_y[i] for i in indices]).to(self.device)
        bl = torch.stack([self.buffer_logits[i] for i in indices]).to(self.device)
        return bx, by, bl

    def train_task(
        self, task_id: int, train_loader: Any, epochs: int = 5
    ) -> Dict[str, Any]:
        self.model.train()
        losses = []
        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
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
                losses.append(loss.item())

        self.update_buffer(train_loader, seen_tasks=task_id + 1)
        return {"task_id": task_id, "loss": sum(losses) / max(len(losses), 1)}


class ERACE:
    """Experience Replay with Asymmetric Cross-Entropy (ER-ACE)."""

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

        self.buffer_x: List[torch.Tensor] = []
        self.buffer_y: List[torch.Tensor] = []
        self.classes_per_task: List[List[int]] = []

    def update_buffer(
        self, train_loader: Any, seen_tasks: int, current_classes: List[int]
    ) -> None:
        collected_x, collected_y = [], []
        for x, y in train_loader:
            collected_x.append(x)
            collected_y.append(y)
        cat_x = torch.cat(collected_x, dim=0)
        cat_y = torch.cat(collected_y, dim=0)

        budget = max(1, self.buffer_size // max(seen_tasks, 1))
        indices = list(range(cat_x.size(0)))
        random.shuffle(indices)
        for idx in indices[:budget]:
            self.buffer_x.append(cat_x[idx].clone())
            self.buffer_y.append(cat_y[idx].clone())
        self.classes_per_task.append(list(current_classes))

        if len(self.buffer_x) > self.buffer_size:
            self.buffer_x = self.buffer_x[-self.buffer_size :]
            self.buffer_y = self.buffer_y[-self.buffer_size :]

    def get_replay_batch(
        self, batch_size: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.buffer_x:
            return None, None
        n = min(batch_size, len(self.buffer_x))
        indices = [random.randint(0, len(self.buffer_x) - 1) for _ in range(n)]
        bx = torch.stack([self.buffer_x[i] for i in indices]).to(self.device)
        by = torch.stack([self.buffer_y[i] for i in indices]).to(self.device)
        return bx, by

    def train_task(
        self,
        task_id: int,
        train_loader: Any,
        epochs: int = 5,
        current_classes: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        if current_classes is None:
            # heuristic default: 10-class problems split into +2 classes per task
            current_classes = [2 * task_id, 2 * task_id + 1]
        self.model.train()
        self.classes_per_task.append(list(current_classes))
        losses = []
        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
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
                losses.append(loss.item())

        self.update_buffer(
            train_loader, seen_tasks=task_id + 1, current_classes=current_classes
        )
        return {"task_id": task_id, "loss": sum(losses) / max(len(losses), 1)}
