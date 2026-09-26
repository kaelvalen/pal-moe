"""Scale-free closed-form ridge with a held-out lambda choice (HEADROOM amendment 3).

The first HEADROOM attempt used a fixed lambda grid {1e-2 .. 1e4}. For a 10,000-d
random-projection readout the Gram diagonal is ~1e6-1e7, so every lambda on that grid
was negligible (held-out accuracy flat, train accuracy 100 %), and ties went to the
smallest lambda. The frozen baselines were silently under-regularised.

This implementation fixes it by construction:

- **Scale normalisation.** The readout input `h` (after any feature map) is divided by
  `s = sqrt(mean ||h||^2)` estimated on the fitting rows; the bias column is appended
  after normalisation. Multiplying the features by any constant leaves the problem
  unchanged. It is bitwise unchanged for powers of two (exact in floating point), and
  unchanged in the selected `c` and in every prediction for other constants (tested).
- **Relative lambda.** `lambda = c * trace(A_feat) / d` on the normalised features,
  `c in {1e-6, ..., 1e1}`.
- **Edge extension.** If the chosen `c` is at an edge of the grid, the grid is extended
  once by three decades in that direction. If it is still at an edge, the arm is flagged
  `converged = False` and the caller withholds the reading.
- **Tie rule with tolerance (a simple 1-SE rule).** Among the `c` whose held-out
  accuracy is within `tie_tol` (0.1 pp) of the best, the largest is chosen.
"""

from __future__ import annotations

import torch

GRID = [10.0**k for k in range(-6, 2)]  # 1e-6 .. 1e1
EXTEND = 3
TIE_TOL = 1e-3  # 0.1 pp of accuracy


def feature_map(z: torch.Tensor, rp: torch.Tensor | None) -> torch.Tensor:
    z = z.double()
    return z if rp is None else torch.relu(z @ rp)


class ScaleFreeRidge:
    """A fitted ridge readout: `predict(z)` applies the same feature map and scale."""

    def __init__(self, W, scale, rp):
        self.W, self.scale, self.rp = W, scale, rp

    def logits(self, z: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        out = []
        for s in range(0, z.size(0), chunk):
            h = feature_map(z[s : s + chunk], self.rp) / self.scale
            h = torch.cat([h, torch.ones(h.size(0), 1, dtype=h.dtype)], 1)
            out.append(h @ self.W)
        return torch.cat(out)

    def accuracy(self, z: torch.Tensor, y: torch.Tensor) -> float:
        if z.size(0) == 0:
            return 0.0
        return float((self.logits(z).argmax(-1) == y).double().mean())


def _scale(Z, rp, chunk=8192) -> float:
    tot, n = 0.0, 0
    for s in range(0, Z.size(0), chunk):
        h = feature_map(Z[s : s + chunk], rp)
        tot += float((h * h).sum())
        n += h.size(0)
    return (tot / max(1, n)) ** 0.5 or 1.0


def _stats(Z, Y, C, rp, scale, chunk=8192):
    A = B = None
    for s in range(0, Z.size(0), chunk):
        h = feature_map(Z[s : s + chunk], rp) / scale
        h = torch.cat([h, torch.ones(h.size(0), 1, dtype=h.dtype)], 1)
        oh = torch.zeros(h.size(0), C, dtype=torch.float64)
        oh[torch.arange(h.size(0)), Y[s : s + chunk]] = 1.0
        A = h.t() @ h if A is None else A + h.t() @ h
        B = h.t() @ oh if B is None else B + h.t() @ oh
    return A, B


def _pick(scores: dict, tie_tol: float) -> float:
    best = max(scores.values())
    return max(c for c, v in scores.items() if v >= best - tie_tol)


def fit_select(
    Ztr: torch.Tensor,
    Ytr: torch.Tensor,
    C: int,
    seed: int,
    rp: torch.Tensor | None = None,
    heldout_frac: float = 0.1,
    tie_tol: float = TIE_TOL,
) -> tuple[ScaleFreeRidge, dict]:
    """Choose `c` on a seeded held-out slice of the TRAIN rows, refit on all of them."""
    n = Ztr.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    k = max(1, int(n * heldout_frac))
    ho, rest = perm[:k], perm[k:]
    scale = _scale(Ztr, rp)  # train rows only
    A_r, B_r = _stats(Ztr[rest], Ytr[rest], C, rp, scale)
    A_h, B_h = _stats(Ztr[ho], Ytr[ho], C, rp, scale)
    d = A_r.size(0) - 1
    unit = float(torch.diagonal(A_r)[:d].sum()) / d
    eye = torch.eye(A_r.size(0), dtype=torch.float64)

    def score(c):  # A_r + c*unit*I is SPD: Cholesky, then two triangular solves
        L = torch.linalg.cholesky(A_r + c * unit * eye)
        W = torch.cholesky_solve(B_r, L)
        return ScaleFreeRidge(W, scale, rp).accuracy(Ztr[ho], Ytr[ho])

    grid = list(GRID)
    scores = {c: score(c) for c in grid}
    c = _pick(scores, tie_tol)
    extended = None
    if c in (min(grid), max(grid)):
        step = 10.0 if c == max(grid) else 0.1
        extra = [c * step**i for i in range(1, EXTEND + 1)]
        scores.update({x: score(x) for x in extra})
        extended = "up" if step > 1 else "down"
        c = _pick(scores, tie_tol)
    all_c = sorted(scores)
    converged = c not in (all_c[0], all_c[-1])
    W = torch.linalg.solve(A_r + A_h + c * unit * eye, B_r + B_h)
    model = ScaleFreeRidge(W, scale, rp)
    return model, {
        "c": c,
        "lambda": c * unit,
        "scale": scale,
        "heldout_acc": {f"{x:.0e}": v for x, v in sorted(scores.items())},
        "extended": extended,
        "converged": converged,
        "train_acc": model.accuracy(Ztr, Ytr),
    }
