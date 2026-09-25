"""
Task-free streaming metrics.

All benchmark numbers in this repo assume the evaluator knows the task
boundaries. A production stream has none, so this module tracks what can be
measured online:

- `online_accuracy`: cumulative accuracy over the whole stream seen so far;
- `recent_accuracy`: accuracy over a sliding window (detects sudden drops);
- `surprise`: exponentially weighted mean NLL (rises on distribution shift);
- `per_domain`: optional per-domain accuracy when the stream carries domain ids.

`StreamingEvaluator` consumes batches in arrival order and never needs task ids.
"""

from collections import defaultdict, deque
from typing import Any, Optional

import torch
import torch.nn.functional as F

__all__ = ["StreamingEvaluator"]


class StreamingEvaluator:
    def __init__(
        self,
        model: Any,
        device: torch.device,
        window: int = 512,
        surprise_momentum: float = 0.05,
    ):
        self.model = model
        self.device = device
        self.window: deque[int] = deque(maxlen=window)
        self.correct = 0
        self.total = 0
        self.surprise = 0.0
        self._surprise_initialised = False
        self.momentum = surprise_momentum
        self.per_domain: dict[int, list] = defaultdict(lambda: [0, 0])

    @torch.no_grad()
    def update(
        self, x: torch.Tensor, y: torch.Tensor, domain: Optional[torch.Tensor] = None
    ) -> None:
        self.model.eval()
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        preds = logits.argmax(dim=-1)
        correct = (preds == y).to(torch.int64)
        self.correct += int(correct.sum().item())
        self.total += int(y.numel())
        self.window.extend(correct.tolist())
        nll = float(F.cross_entropy(logits, y).item())
        if not self._surprise_initialised:
            self.surprise = nll
            self._surprise_initialised = True
        else:
            self.surprise = (1 - self.momentum) * self.surprise + self.momentum * nll
        if domain is not None:
            for d, ok in zip(domain.tolist(), correct.tolist()):
                self.per_domain[int(d)][0] += int(ok)
                self.per_domain[int(d)][1] += 1

    def report(self) -> dict[str, Any]:
        recent = sum(self.window) / len(self.window) if self.window else 0.0
        return {
            "samples_seen": self.total,
            "online_accuracy": self.correct / max(self.total, 1),
            "recent_accuracy": recent,
            "surprise": self.surprise,
            "per_domain_accuracy": {
                str(d): c / max(n, 1) for d, (c, n) in sorted(self.per_domain.items())
            },
        }

    @torch.no_grad()
    def run(self, stream) -> dict[str, Any]:
        """Consumes an iterable of (x, y) or (x, y, domain) batches."""
        for batch in stream:
            if len(batch) == 3:
                x, y, domain = batch
            else:
                x, y = batch
                domain = None
            self.update(x, y, domain)
        return self.report()
