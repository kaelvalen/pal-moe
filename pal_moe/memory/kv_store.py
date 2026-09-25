"""FAST path: an append-only key-value memory with exact delete.

One write is one row: `key = h` (the frozen address), `value` = whatever the
caller wants retrieved (a label, a sentence, a residual delta). Writes are O(1)
appends; `delete` removes the row physically, so the store after
`write(x); delete(x)` is byte-identical to the store before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from pal_moe.address.index import ExactCosineIndex
from pal_moe.core.hashing import digest


@dataclass
class MemoryHit:
    entry_id: str
    score: float
    value: Any
    meta: dict = field(default_factory=dict)


class FastMemory:
    def __init__(self, dim: int, device: str | torch.device = "cpu"):
        self.index = ExactCosineIndex(dim, device=device)
        self._values: dict[str, Any] = {}
        self._meta: dict[str, dict] = {}
        self._raw: dict[
            str, torch.Tensor
        ] = {}  # the key as written (the index normalises)

    def __len__(self) -> int:
        return len(self.index)

    def write(
        self, entry_id: str, key: torch.Tensor, value: Any, meta: dict | None = None
    ) -> None:
        self.index.add(entry_id, key)
        self._raw[entry_id] = key.detach().clone()
        self._values[entry_id] = value
        self._meta[entry_id] = dict(meta or {})

    def delete(self, entry_id: str) -> None:
        self.index.remove(entry_id)
        del self._values[entry_id]
        del self._meta[entry_id]
        del self._raw[entry_id]

    def entry(self, entry_id: str) -> tuple[torch.Tensor, Any, dict]:
        return self._raw[entry_id], self._values[entry_id], self._meta[entry_id]

    def lookup(
        self, query: torch.Tensor, k: int = 1, threshold: float = -1.0
    ) -> list[list[MemoryHit]]:
        """Top-k hits per query row, keeping only those with cosine >= threshold."""
        res = self.index.search(query, k)
        out = []
        for scores, ids in zip(res.scores.tolist(), res.ids):
            out.append(
                [
                    MemoryHit(i, s, self._values[i], self._meta[i])
                    for s, i in zip(scores, ids)
                    if s >= threshold
                ]
            )
        return out

    def state_digest(self) -> str:
        values = [repr(self._values[i]) for i in self.index.ids]
        return digest(self.index.ids, self.index.keys, values)

    def parameter_count(self) -> int:
        return 0
