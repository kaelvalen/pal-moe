"""MEDIUM path: closed-form linear edits from additive float64 sufficient statistics.

For a linear map fitted by ridge regression, `W = B^T (lambda I + A)^-1` with
`A = sum K^T K` and `B = sum K^T V`. Both sums are additive over edits, so

- learning a batch is adding its contribution (closed form, no optimiser),
- the result is independent of the order batches arrived in, and
- forgetting a batch is removing its contribution (exact unlearning).

Two floating-point facts decide how this is implemented:

1. **float64.** E-TID2 (G3) measured max|dW| = 1.26e-3 between continual and one-shot
   ridge in float32 (argmax still identical). Statistics here are float64.
2. **Canonical summation order.** Floating-point addition is not associative, so a
   running sum is only *approximately* order-invariant and `(A + dA) - dA` is only
   approximately `A`. The solved weights therefore come from the contributions summed
   in a canonical order (sorted by content hash, then id). That makes order
   invariance and reversibility **bitwise**. The arrival-order running sum is kept
   alongside so the order-invariance guard can report how far the naive running sum
   drifts (max|dW| canonical vs running).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pal_moe.core.hashing import digest


@dataclass
class Contribution:
    edit_id: str
    content_hash: str
    dA: torch.Tensor  # [d', d'] float64
    dB: torch.Tensor  # [d', m]  float64
    n: int


class LinearStats:
    """Additive statistics of a ridge-fitted linear map `K (d) -> V (m)`.

    `bias=True` appends a ones column (the intercept is solved jointly), matching
    `pal_moe.arch.readouts.RidgeReadout` term for term: `A_0 = lambda I` on the
    augmented dimension.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        ridge: float = 1.0,
        bias: bool = True,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.dim, self.out_dim, self.ridge, self.bias = (
            int(dim),
            int(out_dim),
            float(ridge),
            bool(bias),
        )
        self.device, self.dtype = torch.device(device), dtype
        d1 = self.dim + int(self.bias)
        self.A0 = self.ridge * torch.eye(d1, device=self.device, dtype=dtype)
        self._contrib: dict[str, Contribution] = {}
        self._arrival: list[str] = []
        self.A_run = self.A0.clone()
        self.B_run = torch.zeros(d1, self.out_dim, device=self.device, dtype=dtype)
        self._W: torch.Tensor | None = None  # canonical solution cache

    # -- contributions ---------------------------------------------------

    def _augment(self, k: torch.Tensor) -> torch.Tensor:
        k = k.to(self.device, self.dtype)
        if not self.bias:
            return k
        return torch.cat(
            [k, torch.ones(k.size(0), 1, device=k.device, dtype=k.dtype)], dim=1
        )

    @torch.no_grad()
    def contribution(
        self, edit_id: str, K: torch.Tensor, V: torch.Tensor
    ) -> Contribution:
        Ka = self._augment(K.detach().reshape(-1, self.dim))
        V = V.detach().to(self.device, self.dtype).reshape(Ka.size(0), self.out_dim)
        return Contribution(
            edit_id, digest(K.detach(), V.detach()), Ka.t() @ Ka, Ka.t() @ V, Ka.size(0)
        )

    @torch.no_grad()
    def add(self, contribution: Contribution) -> None:
        if contribution.edit_id in self._contrib:
            raise KeyError(f"duplicate edit id {contribution.edit_id!r}")
        self._contrib[contribution.edit_id] = contribution
        self._arrival.append(contribution.edit_id)
        self.A_run += contribution.dA
        self.B_run += contribution.dB
        self._W = None

    @torch.no_grad()
    def remove(self, edit_id: str) -> Contribution:
        c = self._contrib.pop(edit_id)
        self._arrival.remove(edit_id)
        self.A_run -= c.dA  # the naive downdate, kept only for the drift report
        self.B_run -= c.dB
        self._W = None
        return c

    def __contains__(self, edit_id: str) -> bool:
        return edit_id in self._contrib

    @property
    def edit_ids(self) -> list[str]:
        return list(self._arrival)

    @property
    def n_samples(self) -> int:
        return sum(c.n for c in self._contrib.values())

    # -- solutions -------------------------------------------------------

    def canonical_order(self) -> list[str]:
        return [
            c.edit_id
            for c in sorted(
                self._contrib.values(), key=lambda c: (c.content_hash, c.edit_id)
            )
        ]

    @torch.no_grad()
    def canonical_sums(self) -> tuple[torch.Tensor, torch.Tensor]:
        A = self.A0.clone()
        B = torch.zeros_like(self.B_run)
        for eid in self.canonical_order():
            A += self._contrib[eid].dA
            B += self._contrib[eid].dB
        return A, B

    @torch.no_grad()
    def solve(self) -> torch.Tensor:
        """`W` [m, d'] from the canonical sums (cached until the next add/remove)."""
        if self._W is None:
            A, B = self.canonical_sums()
            self._W = torch.linalg.solve(A, B).t().contiguous()
        return self._W

    @torch.no_grad()
    def solve_running(self) -> torch.Tensor:
        return torch.linalg.solve(self.A_run, self.B_run).t().contiguous()

    @torch.no_grad()
    def predict(self, K: torch.Tensor) -> torch.Tensor:
        return self._augment(K.reshape(-1, self.dim)) @ self.solve().t()

    def state_digest(self) -> str:
        A, B = self.canonical_sums()
        return digest(A, B)

    def order_report(self, canary: torch.Tensor | None = None) -> dict:
        """Canonical vs arrival-order running solution: the order-invariance guard."""
        if not self._contrib:
            return {"max_abs_dW": 0.0, "argmax_identical": True, "n_edits": 0}
        Wc, Wr = self.solve(), self.solve_running()
        out = {
            "max_abs_dW": float((Wc - Wr).abs().max()),
            "n_edits": len(self._contrib),
        }
        if canary is not None:
            Ka = self._augment(canary.reshape(-1, self.dim))
            out["argmax_identical"] = bool(
                torch.equal((Ka @ Wc.t()).argmax(-1), (Ka @ Wr.t()).argmax(-1))
            )
        return out

    def parameter_count(self) -> int:
        return 0  # statistics are buffers, never optimised


def one_hot(y: torch.Tensor, num_classes: int, dtype=torch.float64) -> torch.Tensor:
    out = torch.zeros(y.numel(), num_classes, dtype=dtype, device=y.device)
    out[torch.arange(y.numel(), device=y.device), y.reshape(-1).long()] = 1.0
    return out
