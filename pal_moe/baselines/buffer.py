"""
Rehearsal buffers for the memory-based baselines.

Two policies:
- "recency": append a per-task budget of randomly chosen samples and drop the
  oldest whenever the capacity is exceeded. This is the original behaviour and
  is kept as the default so published numbers stay reproducible.
- "reservoir": classic uniform reservoir sampling over the whole stream, so
  early tasks are not evicted in favour of recent ones (the recency policy
  over-represents the newest tasks once the buffer saturates).

Both policies draw samples in the same order as the original inline code, so
"recency" is bit-for-bit identical to the previous implementation.
"""

import random
from typing import Optional

import torch

BUFFER_MODES = ("recency", "reservoir")


class SampleBuffer:
    """Stores (x, y[, logits]) exemplars up to `capacity` according to `mode`."""

    def __init__(self, capacity: int, mode: str = "recency"):
        if mode not in BUFFER_MODES:
            raise ValueError(f"unknown buffer mode {mode!r}; use one of {BUFFER_MODES}")
        self.capacity = int(capacity)
        self.mode = mode
        self.x: list[torch.Tensor] = []
        self.y: list[torch.Tensor] = []
        self.logits: list[torch.Tensor] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self.x)

    def _append(self, x, y, logits) -> None:
        self.x.append(x.clone())
        self.y.append(y.clone())
        if logits is not None:
            self.logits.append(logits.clone())

    def add_task(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        seen_tasks: int = 1,
        per_task_budget: Optional[int] = None,
    ) -> None:
        """
        Adds samples from one task.

        Recency: keeps `per_task_budget` (default capacity // seen_tasks)
        randomly chosen samples and trims the oldest overflow.
        Reservoir: uniform replacement over the whole stream (no trimming).
        """
        n = x.size(0)
        if self.mode == "recency":
            budget = (
                per_task_budget
                if per_task_budget is not None
                else max(1, self.capacity // max(seen_tasks, 1))
            )
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices[:budget]:
                self._append(x[i], y[i], None if logits is None else logits[i])
            self.seen += n
            if len(self.x) > self.capacity:
                self.x = self.x[-self.capacity :]
                self.y = self.y[-self.capacity :]
                self.logits = self.logits[-self.capacity :]
        else:
            for i in range(n):
                self.seen += 1
                item = (x[i], y[i], None if logits is None else logits[i])
                if len(self.x) < self.capacity:
                    self._append(*item)
                else:
                    j = random.randrange(self.seen)
                    if j < self.capacity:
                        self.x[j] = item[0].clone()
                        self.y[j] = item[1].clone()
                        if item[2] is not None:
                            self.logits[j] = item[2].clone()

    def sample_tensors(
        self, k: int, device: torch.device
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Random batch of up to k exemplars on `device` (logits if stored)."""
        if not self.x:
            return None, None, None
        n = min(k, len(self.x))
        indices = [random.randint(0, len(self.x) - 1) for _ in range(n)]
        bx = torch.stack([self.x[i] for i in indices]).to(device)
        by = torch.stack([self.y[i] for i in indices]).to(device)
        bl = None
        if self.logits:
            bl = torch.stack([self.logits[i] for i in indices]).to(device)
        return bx, by, bl

    def memory_bytes(self) -> int:
        """Bytes actually held by the stored exemplars (x, y and optional logits)."""
        total = 0
        for tensor in self.x + self.y + self.logits:
            total += tensor.numel() * tensor.element_size()
        return int(total)


def task_class_counts(buffer: SampleBuffer, num_classes: int) -> list[int]:
    """Debug helper: class histogram of the stored labels."""
    counts = [0] * num_classes
    for label in buffer.y:
        counts[int(label.item())] += 1
    return counts
