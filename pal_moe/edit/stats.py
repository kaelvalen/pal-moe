"""MEDIUM path: closed-form linear edits from additive float64 sufficient statistics.

For a linear map fitted by ridge regression, `W = B^T (lambda I + A)^-1` with
`A = sum K^T K` and `B = sum K^T V`. Both sums are additive over edits, so

- learning a batch is adding its contribution to ONE accumulator (closed form),
- the result does not depend on the order batches arrived in (up to fp rounding),
- forgetting a batch is subtracting its contribution (a downdate: exact unlearning in
  exact arithmetic, fp-rounding-close in float64).

Storage. The accumulators are one `[d', d']` and one `[d', m]` float64 matrix,
independent of the number of edits. To be able to subtract an edit later, each edit
keeps the *smaller* of its two representations: the raw factors `(K, V)` when
`n <= d'` (a single fact, a small batch: `n x d'`), else its `(dA, dB)` (`d' x d'`).
Recomputing `dA = K^T K` from stored factors is deterministic, so the subtraction
removes exactly the bits that were added.

What the guards measure (they are measurements, not consequences of a design
choice): `order_report` re-sums the live contributions in a seeded random
permutation and reports max|dW| against the accumulator; `recompute_report` re-sums
them in arrival order and reports max|dW| after downdates. Both are compared with a
tolerance. Bitwise equality is only claimed where it is real: when the last edit is
forgotten the accumulators are reset to `(lambda I, 0)` exactly.

E-TID2 (G3) measured max|dW| = 1.26e-3 between continual and one-shot ridge in
float32; hence float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pal_moe.core.hashing import digest


@dataclass
class Contribution:
    edit_id: str
    content_hash: str
    n: int
    K: torch.Tensor | None = None  # [n, d'] augmented keys, float64 (when n <= d')
    V: torch.Tensor | None = None  # [n, m] float64
    dA: torch.Tensor | None = None  # [d', d'] (when n > d')
    dB: torch.Tensor | None = None  # [d', m]

    def delta(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dA is not None:
            return self.dA, self.dB
        return self.K.t() @ self.K, self.K.t() @ self.V

    def nbytes(self) -> int:
        ts = [t for t in (self.K, self.V, self.dA, self.dB) if t is not None]
        return sum(t.numel() * t.element_size() for t in ts)


class LinearStats:
    """Additive statistics of a ridge-fitted linear map `K (d) -> V (m)`.

    `bias=True` appends a ones column (intercept solved jointly), matching
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
        self.dim, self.out_dim = int(dim), int(out_dim)
        self.ridge, self.bias = float(ridge), bool(bias)
        self.device, self.dtype = torch.device(device), dtype
        self.d1 = self.dim + int(self.bias)
        self.A0 = self.ridge * torch.eye(self.d1, device=self.device, dtype=dtype)
        self.A = self.A0.clone()
        self.B = torch.zeros(self.d1, self.out_dim, device=self.device, dtype=dtype)
        self._contrib: dict[str, Contribution] = {}
        self._arrival: list[str] = []
        self._W: torch.Tensor | None = None

    # -- contributions ---------------------------------------------------

    def _augment(self, k: torch.Tensor) -> torch.Tensor:
        k = k.to(self.device, self.dtype)
        if not self.bias:
            return k
        ones = torch.ones(k.size(0), 1, device=k.device, dtype=k.dtype)
        return torch.cat([k, ones], dim=1)

    @torch.no_grad()
    def contribution(
        self, edit_id: str, K: torch.Tensor, V: torch.Tensor
    ) -> Contribution:
        Ka = self._augment(K.detach().reshape(-1, self.dim))
        V64 = V.detach().to(self.device, self.dtype).reshape(Ka.size(0), self.out_dim)
        h = digest(K.detach(), V.detach())
        if Ka.size(0) <= self.d1:
            return Contribution(edit_id, h, Ka.size(0), K=Ka, V=V64)
        return Contribution(edit_id, h, Ka.size(0), dA=Ka.t() @ Ka, dB=Ka.t() @ V64)

    @torch.no_grad()
    def add(self, c: Contribution) -> None:
        if c.edit_id in self._contrib:
            raise KeyError(f"duplicate edit id {c.edit_id!r}")
        dA, dB = c.delta()
        self.A += dA
        self.B += dB
        self._contrib[c.edit_id] = c
        self._arrival.append(c.edit_id)
        self._W = None

    @torch.no_grad()
    def remove(self, edit_id: str) -> Contribution:
        """Downdate: subtract the edit's contribution from the accumulators."""
        c = self._contrib.pop(edit_id)
        self._arrival.remove(edit_id)
        if not self._contrib:  # the only exact statement available: back to the prior
            self.A = self.A0.clone()
            self.B.zero_()
        else:
            dA, dB = c.delta()
            self.A -= dA
            self.B -= dB
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

    def storage_bytes(self) -> dict:
        acc = (self.A.numel() + self.B.numel()) * self.A.element_size()
        per = sum(c.nbytes() for c in self._contrib.values())
        return {
            "accumulators": acc,
            "per_edit_total": per,
            "n_edits": len(self._contrib),
        }

    # -- solutions -------------------------------------------------------

    @torch.no_grad()
    def solve(self) -> torch.Tensor:
        """`W` [m, d'] from the accumulators (cached until the next add/remove)."""
        if self._W is None:
            self._W = torch.linalg.solve(self.A, self.B).t().contiguous()
        return self._W

    @torch.no_grad()
    def resum(self, order: list[str]) -> torch.Tensor:
        """Reference solution: the live contributions re-summed in `order`."""
        A, B = self.A0.clone(), torch.zeros_like(self.B)
        for eid in order:
            dA, dB = self._contrib[eid].delta()
            A += dA
            B += dB
        return torch.linalg.solve(A, B).t().contiguous()

    @torch.no_grad()
    def predict(self, K: torch.Tensor) -> torch.Tensor:
        return self._augment(K.reshape(-1, self.dim)) @ self.solve().t()

    def state_digest(self) -> str:
        return digest(self.A, self.B)

    def _argmax_identical(self, Wa, Wb, canary) -> bool | None:
        if canary is None:
            return None
        Ka = self._augment(canary.reshape(-1, self.dim))
        return bool(torch.equal((Ka @ Wa.t()).argmax(-1), (Ka @ Wb.t()).argmax(-1)))

    @torch.no_grad()
    def order_report(self, canary=None, tol: float = 1e-10, seed: int = 0) -> dict:
        """Accumulator vs the live contributions re-summed in a random permutation."""
        n = len(self._arrival)
        if n == 0:
            return {"n_edits": 0, "max_abs_dW_permutation": 0.0, "pass": True}
        g = torch.Generator().manual_seed(seed + n)
        perm = [self._arrival[i] for i in torch.randperm(n, generator=g).tolist()]
        Wp = self.resum(perm)
        W = self.solve()
        rep = {
            "n_edits": n,
            "permutation": perm if n <= 32 else None,
            "max_abs_dW_permutation": float((W - Wp).abs().max()),
            "tolerance": tol,
            "argmax_identical": self._argmax_identical(W, Wp, canary),
        }
        rep["pass"] = (
            rep["max_abs_dW_permutation"] <= tol
            and rep["argmax_identical"] is not False
        )
        return rep

    @torch.no_grad()
    def recompute_report(self, canary=None, tol: float = 1e-10) -> dict:
        """Accumulator (after any downdates) vs the live contributions re-summed."""
        if not self._arrival:
            exact = torch.equal(self.A, self.A0) and not bool(self.B.any())
            return {
                "n_edits": 0,
                "max_abs_dW_recompute": 0.0,
                "bitwise_prior": exact,
                "pass": exact,
            }
        Wr = self.resum(self._arrival)
        W = self.solve()
        rep = {
            "n_edits": len(self._arrival),
            "max_abs_dW_recompute": float((W - Wr).abs().max()),
            "tolerance": tol,
            "argmax_identical": self._argmax_identical(W, Wr, canary),
        }
        rep["pass"] = (
            rep["max_abs_dW_recompute"] <= tol and rep["argmax_identical"] is not False
        )
        return rep

    def parameter_count(self) -> int:
        return 0  # statistics are buffers, never optimised


def one_hot(y: torch.Tensor, num_classes: int, dtype=torch.float64) -> torch.Tensor:
    out = torch.zeros(y.numel(), num_classes, dtype=dtype, device=y.device)
    out[torch.arange(y.numel(), device=y.device), y.reshape(-1).long()] = 1.0
    return out
