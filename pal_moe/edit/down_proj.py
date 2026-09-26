"""MEDIUM path for a language model: a closed-form edit of one MLP down-projection.

Keys `k` are the down-projection's inputs at the edited token, values `v*` the
outputs that make the target likely (`HFCausalLM.target_value`), residuals
`R = V* - K W^T` computed against the **base** model (so contributions are
independent of each other). The MEMIT least-squares edit is

    Delta = (C0 + S)^-1 B,     S = sum K^T K,  B = sum K^T R        [d_ff, d_model]

applied as `y += x @ Delta`.

**The prior `C0` must come from a corpus, not from the edit keys.** MEMIT's locality
comes from `C0 = lambda * E[k k^T]` over keys of text the model already handles
(Wikipedia in MEMIT); it penalises moving any direction those keys occupy. A prior
built from the edit keys alone (or `eps * I`) leaves every non-edit key unprotected.
`estimate_key_covariance` takes keys collected over **all token positions** of a
corpus (`HFCausalLM.collect_keys`), and the editor records the prior's hash and the
number of tokens behind it; it refuses a prior estimated from fewer tokens than
`d_ff` (rank-deficient: most directions unprotected).

Two solvers, the same statement:

- `mode="accumulate"` (default): `S` and `B` are single accumulators (float64, one
  `d_ff x d_ff` matrix regardless of the number of edits); forget subtracts the edit's
  `K^T K`, `K^T R` (a downdate). Each edit keeps its `(K, R)` rows (`n x d_ff`), the
  minimum needed to subtract it. Each solve is `O(d_ff^3)`.
- `mode="woodbury"`: no `d_ff^2` accumulator; `Delta = C0^-1 K^T (I + K C0^-1 K^T)^-1 R`
  over the stacked rows (arrival order), `O(n^2 d_ff)` per solve. Forget removes the
  rows. For large `d_ff` with few edits.

Guards measure, they do not assume: `order_report` re-solves with the edits in a seeded
random permutation, `recompute_report` re-solves from scratch after downdates; both
report max|dDelta| against a tolerance. When the last edit is forgotten the delta is
`None` and the hook is removed, so the base model is restored bitwise.
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


@dataclass
class KeyPrior:
    cov: torch.Tensor  # [d_ff, d_ff] float64
    n_tokens: int
    source: str
    hash: str


def estimate_key_covariance(
    keys: torch.Tensor, weight: float = 15000.0, ridge: float = 1e-4, source: str = ""
) -> KeyPrior:
    """`C0 = weight * E[k k^T] + ridge * I` (float64) from [N, d_ff] corpus keys."""
    k = keys.double()
    C = weight * (k.t() @ k) / max(1, k.size(0))
    C = C + ridge * torch.eye(C.size(0), dtype=C.dtype, device=C.device)
    return KeyPrior(C, int(k.size(0)), source, digest(C))


class DownProjEdit:
    def __init__(
        self, prior: KeyPrior, mode: str = "accumulate", min_tokens: int | None = None
    ):
        if not isinstance(prior, KeyPrior):
            raise TypeError(
                "prior must be a KeyPrior from estimate_key_covariance(corpus keys)"
            )
        d_ff = prior.cov.size(0)
        need = d_ff if min_tokens is None else min_tokens
        if prior.n_tokens < need:
            raise ValueError(
                f"prior estimated from {prior.n_tokens} tokens < {need}: rank-deficient, it "
                "would leave most key directions unprotected; collect more corpus tokens"
            )
        if mode not in ("accumulate", "woodbury"):
            raise KeyError(mode)
        self.prior, self.mode = prior, mode
        self.C0 = prior.cov.double()
        self.C0_inv = torch.linalg.inv(self.C0) if mode == "woodbury" else None
        self.S = torch.zeros_like(self.C0) if mode == "accumulate" else None
        self.B: torch.Tensor | None = None
        self._contrib: dict[str, KeyValueContribution] = {}
        self._arrival: list[str] = []
        self._delta: torch.Tensor | None = None

    def contribution(
        self, edit_id: str, K: torch.Tensor, R: torch.Tensor
    ) -> KeyValueContribution:
        K = K.detach().double().to(self.C0.device)
        R = R.detach().double().to(self.C0.device)
        return KeyValueContribution(edit_id, digest(K, R), K, R)

    def add(self, c: KeyValueContribution) -> None:
        if c.edit_id in self._contrib:
            raise KeyError(f"duplicate edit id {c.edit_id!r}")
        if self.mode == "accumulate":
            if self.B is None:
                self.B = torch.zeros(
                    self.C0.size(0),
                    c.R.size(1),
                    dtype=self.C0.dtype,
                    device=self.C0.device,
                )
            self.S += c.K.t() @ c.K
            self.B += c.K.t() @ c.R
        self._contrib[c.edit_id] = c
        self._arrival.append(c.edit_id)
        self._delta = None

    def remove(self, edit_id: str) -> KeyValueContribution:
        c = self._contrib.pop(edit_id)
        self._arrival.remove(edit_id)
        if self.mode == "accumulate":
            if not self._contrib:
                self.S.zero_()
                self.B = None
            else:
                self.S -= c.K.t() @ c.K
                self.B -= c.K.t() @ c.R
        self._delta = None
        return c

    # -- solves ---------------------------------------------------------------------

    def _solve_rows(self, order: list[str]) -> torch.Tensor | None:
        if not order:
            return None
        K = torch.cat([self._contrib[i].K for i in order])
        R = torch.cat([self._contrib[i].R for i in order])
        if self.mode == "woodbury":
            CK = self.C0_inv @ K.t()
            inner = torch.eye(K.size(0), dtype=K.dtype, device=K.device) + K @ CK
            return CK @ torch.linalg.solve(inner, R)
        S = torch.zeros_like(self.C0)
        B = torch.zeros(self.C0.size(0), R.size(1), dtype=R.dtype, device=R.device)
        for i in order:
            S += self._contrib[i].K.t() @ self._contrib[i].K
            B += self._contrib[i].K.t() @ self._contrib[i].R
        return torch.linalg.solve(self.C0 + S, B)

    def solve(self) -> torch.Tensor | None:
        if not self._contrib:
            return None
        if self._delta is None:
            if self.mode == "accumulate":
                self._delta = torch.linalg.solve(self.C0 + self.S, self.B)
            else:
                self._delta = self._solve_rows(self._arrival)
        return self._delta

    def order_report(self, tol: float = 1e-8, seed: int = 0) -> dict:
        n = len(self._arrival)
        if n == 0:
            return {"n_edits": 0, "max_abs_dDelta_permutation": 0.0, "pass": True}
        g = torch.Generator().manual_seed(seed + n)
        perm = [self._arrival[i] for i in torch.randperm(n, generator=g).tolist()]
        d = float((self.solve() - self._solve_rows(perm)).abs().max())
        return {
            "n_edits": n,
            "max_abs_dDelta_permutation": d,
            "tolerance": tol,
            "pass": d <= tol,
        }

    def recompute_report(self, tol: float = 1e-8) -> dict:
        if not self._arrival:
            exact = self.S is None or not bool(self.S.any())
            return {
                "n_edits": 0,
                "max_abs_dDelta_recompute": 0.0,
                "bitwise_prior": exact,
                "pass": exact,
            }
        d = float((self.solve() - self._solve_rows(self._arrival)).abs().max())
        return {
            "n_edits": len(self._arrival),
            "max_abs_dDelta_recompute": d,
            "tolerance": tol,
            "pass": d <= tol,
        }

    def storage_bytes(self) -> dict:
        acc = (
            0
            if self.S is None
            else self.S.numel() * 8 + (0 if self.B is None else self.B.numel() * 8)
        )
        per = sum((c.K.numel() + c.R.numel()) * 8 for c in self._contrib.values())
        return {
            "accumulators": acc,
            "per_edit_total": per,
            "n_edits": len(self._contrib),
        }

    def state_digest(self) -> str:
        return digest(
            self._arrival, self.S if self.S is not None else "woodbury", self.B
        )

    def parameter_count(self) -> int:
        return 0
