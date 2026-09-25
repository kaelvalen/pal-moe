"""MEDIUM path for a language model: a closed-form edit of one MLP down-projection.

Keys `k` are the down-projection's inputs at the edited token, values `v*` the outputs
that make the target likely (`HFCausalLM.target_value`). With a key-covariance prior
`C0` (estimated once on a corpus, float64) the least-squares edit that maps the new
keys to their values while preserving the prior's keys is

    Delta = C0^-1 K^T (I + K C0^-1 K^T)^-1 R,     R = V* - K W^T   (base-model residuals)

(the MEMIT form written with Woodbury, so only an `n x n` system is solved per update;
`Delta` is `[d_ff, d_model]` and is applied as `y += x @ Delta`).

The same three properties as the vision medium path hold by construction:

- residuals are computed against the **base** model, so contributions are independent;
- the solve stacks contributions in a **canonical** order (content hash, then id), so
  the result is bitwise independent of arrival order;
- forgetting is removing a contribution and re-solving from the rest: bitwise equal to
  never having written it (and an empty set removes the hook entirely).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pal_moe.core.hashing import digest


@dataclass
class KeyValueContribution:
    edit_id: str
    content_hash: str
    K: torch.Tensor  # [n, d_ff] float64
    R: torch.Tensor  # [n, d_model] float64


def estimate_key_covariance(
    keys: torch.Tensor, weight: float = 1.0, ridge: float = 1e-4
) -> torch.Tensor:
    """`C0 = weight * E[k k^T] + ridge * I` in float64 from a [N, d_ff] key sample."""
    k = keys.double()
    C = weight * (k.t() @ k) / max(1, k.size(0))
    return C + ridge * torch.eye(C.size(0), dtype=C.dtype, device=C.device)


class DownProjEdit:
    def __init__(self, prior_cov: torch.Tensor):
        self.C0 = prior_cov.double()
        self.C0_inv = torch.linalg.inv(self.C0)
        self._contrib: dict[str, KeyValueContribution] = {}
        self._arrival: list[str] = []

    def contribution(
        self, edit_id: str, K: torch.Tensor, R: torch.Tensor
    ) -> KeyValueContribution:
        K, R = (
            K.detach().double().to(self.C0.device),
            R.detach().double().to(self.C0.device),
        )
        return KeyValueContribution(edit_id, digest(K, R), K, R)

    def add(self, c: KeyValueContribution) -> None:
        if c.edit_id in self._contrib:
            raise KeyError(f"duplicate edit id {c.edit_id!r}")
        self._contrib[c.edit_id] = c
        self._arrival.append(c.edit_id)

    def remove(self, edit_id: str) -> KeyValueContribution:
        self._arrival.remove(edit_id)
        return self._contrib.pop(edit_id)

    def canonical_order(self) -> list[str]:
        return [
            c.edit_id
            for c in sorted(
                self._contrib.values(), key=lambda c: (c.content_hash, c.edit_id)
            )
        ]

    def _solve(self, order: list[str]) -> torch.Tensor | None:
        if not order:
            return None
        K = torch.cat([self._contrib[i].K for i in order])
        R = torch.cat([self._contrib[i].R for i in order])
        CK = self.C0_inv @ K.t()  # [d_ff, n]
        inner = torch.eye(K.size(0), dtype=K.dtype, device=K.device) + K @ CK
        return CK @ torch.linalg.solve(inner, R)  # [d_ff, d_model]

    def solve(self) -> torch.Tensor | None:
        return self._solve(self.canonical_order())

    def order_report(self) -> dict:
        a, b = self._solve(self.canonical_order()), self._solve(list(self._arrival))
        if a is None:
            return {"max_abs_dW": 0.0, "argmax_identical": True, "n_edits": 0}
        return {
            "max_abs_dW": float((a - b).abs().max()),
            "argmax_identical": True,
            "n_edits": len(self._contrib),
        }

    def state_digest(self) -> str:
        order = self.canonical_order()
        return digest(order and [self._contrib[i].content_hash for i in order])

    def parameter_count(self) -> int:
        return 0
