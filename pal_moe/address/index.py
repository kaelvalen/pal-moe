"""Parameter-free retrieval over frozen keys.

`ExactCosineIndex` is the reference implementation: brute-force cosine top-k over
every stored key. An ANN index can replace it later behind the same three calls
(`add`, `remove`, `search`) as long as it reproduces the exact index's top-k on the
anchor sets. No index holds a trainable parameter (AC3: a fixed retrieval address
is a drop-in for the learned one; C1/AC1: learned addresses drift).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SearchResult:
    scores: torch.Tensor  # [B, k] cosine similarity, best first
    ids: list[list[str]]  # [B][k] entry ids (fewer than k if the index is small)


class ExactCosineIndex:
    """Insertion-ordered key store with exact cosine search.

    Keys are stored L2-normalised in float32. Removal is physical (the row is
    deleted), so an index after `add(x); remove(x)` is byte-identical to the index
    before - the property the fast path's reversibility guard rests on.
    """

    def __init__(self, dim: int, device: str | torch.device = "cpu"):
        self.dim = int(dim)
        self.device = torch.device(device)
        self._keys = torch.zeros(0, self.dim, device=self.device)
        self._ids: list[str] = []

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    @property
    def keys(self) -> torch.Tensor:
        return self._keys

    @torch.no_grad()
    def add(self, entry_id: str, key: torch.Tensor) -> None:
        if entry_id in self._ids:
            raise KeyError(f"duplicate id {entry_id!r}")
        key = F.normalize(key.detach().float().reshape(1, -1).to(self.device), dim=-1)
        if key.size(1) != self.dim:
            raise ValueError(f"key dim {key.size(1)} != index dim {self.dim}")
        self._keys = torch.cat([self._keys, key], dim=0)
        self._ids.append(entry_id)

    @torch.no_grad()
    def remove(self, entry_id: str) -> None:
        row = self._ids.index(entry_id)
        keep = [i for i in range(len(self._ids)) if i != row]
        self._keys = self._keys[keep].contiguous() if keep else self._keys[:0]
        del self._ids[row]

    @torch.no_grad()
    def search(self, query: torch.Tensor, k: int = 1) -> SearchResult:
        query = F.normalize(
            query.detach().float().reshape(-1, self.dim).to(self.device), dim=-1
        )
        if not self._ids:
            return SearchResult(
                torch.zeros(query.size(0), 0, device=self.device),
                [[] for _ in range(query.size(0))],
            )
        sim = query @ self._keys.t()
        values, rows = sim.topk(min(k, len(self._ids)), dim=-1)
        ids = [[self._ids[r] for r in row] for row in rows.tolist()]
        return SearchResult(values, ids)

    def parameter_count(self) -> int:
        return 0
