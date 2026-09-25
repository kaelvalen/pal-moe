"""`ridge_class`: the E-TID2 router - continual class-level ridge, task = owner of argmax class.

The router holds no statistics of its own: it reads the medium path's `LinearStats`
(float64, additive, subtractable) and a class -> task ownership map. Its task score
is the max ridge logit over the task's classes; unowned tasks score -1e9. This is the
rule E-TID2 measured term for term (`scatter_reduce(..., "amax")`), with the ridge
solved in float64 instead of float32.
"""

from __future__ import annotations

import torch

from pal_moe.edit.stats import LinearStats


class RidgeClassRouter:
    name = "ridge_class"

    def __init__(
        self,
        stats: LinearStats,
        class_owner: dict[int, int],
        num_tasks: int | None = None,
    ):
        self.stats = stats
        self.class_owner = class_owner  # shared, updated by the facade
        self._num_tasks = num_tasks

    @property
    def num_tasks(self) -> int:
        if self._num_tasks is not None:
            return self._num_tasks
        return (max(self.class_owner.values()) + 1) if self.class_owner else 1

    @torch.no_grad()
    def class_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.stats.predict(z)

    @torch.no_grad()
    def scores(
        self, z: torch.Tensor, class_logits: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = self.class_logits(z) if class_logits is None else class_logits
        C = logits.size(1)
        owner = torch.full((C,), -1, dtype=torch.long, device=logits.device)
        for c, t in self.class_owner.items():
            owner[c] = t
        known = owner >= 0
        out = torch.full(
            (logits.size(0), self.num_tasks),
            -1e9,
            device=logits.device,
            dtype=logits.dtype,
        )
        if bool(known.any()):
            out.scatter_reduce_(
                1, owner[known].expand(logits.size(0), -1), logits[:, known], "amax"
            )
        return out

    @torch.no_grad()
    def top_k(
        self, z: torch.Tensor, k: int = 1, class_logits: torch.Tensor | None = None
    ):
        s = self.scores(z, class_logits)
        values, ids = s.topk(min(k, s.size(1)), dim=-1)
        return ids, values

    def parameter_count(self) -> int:
        return 0
