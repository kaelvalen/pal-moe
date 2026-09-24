"""
Readout implementations for the S1 contract: `Z -> Y`.

Readouts are where E0 found the first surprise: a training-free prototype
readout (NCM, 0 parameters, 307 KB) beat the trained classifier of the whole v1
system (70.34 vs 59.30 average accuracy on CIFAR-100/ViT). That is why the
readout is a separate axis and not a detail of the expert.

Registered names:

    ncm        nearest class mean, training-free, incremental running means
    cosine     `scale * <normalize(z), normalize(W)>`, trained by Adam
    linear     plain linear layer, trained by Adam
    logistic   linear + cross-entropy over the *seen* classes, frozen old rows
    ridge      closed-form ridge regression on one-hot targets (no optimizer)
    mlp        a small MLP, trained by Adam

All of them expose the same `fit(z, y, seen_classes)` / `predict(z)` contract,
so a study swaps one for another without touching the training loop. Every
`fit` accepts a single batch or a whole task; batching is the caller's choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_readout

__all__ = [
    "NCMReadout",
    "CosineReadout",
    "LinearReadout",
    "RidgeReadout",
    "MLPReadout",
    "mask_unseen",
]


def trainable_hook(readout, active_classes: list[int]):
    """Ask a readout which of its class rows may move NOW; no-op when it has none.

    `active_classes` is the set whose parameters are allowed to change in this
    update - the current task's classes under the sequential constraint, or all
    seen classes when the readout is being refitted jointly. Passing "all seen
    classes" instead is the classic mistake: it re-opens every old row and the
    readout collapses to 7.69% (forgetting 92.35%) on CIFAR-100/ViT.
    """
    hook = getattr(readout, "trainable_hook", None)
    return hook(list(active_classes)) if hook is not None else _NullHandle()


def mask_unseen(logits: torch.Tensor, seen_classes: list[int]) -> torch.Tensor:
    """`-inf` on classes not seen yet: the class-incremental convention.

    Shared by every readout so that a study cannot accidentally compare a
    masked method against an unmasked one. E0 measured that this convention
    matters: a growing head faces `5*(t-1)` distractors where the v1 fixed
    100-way head faces 95.
    """
    if not seen_classes or len(seen_classes) == logits.size(-1):
        return logits
    unseen = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
    unseen[torch.tensor(seen_classes, device=logits.device, dtype=torch.long)] = False
    return logits.masked_fill(unseen, -1e9)


class _NullHandle:
    """No-op gradient-hook handle, so the loop can call `trainable_hook`
    unconditionally on every readout."""

    def remove(self) -> None:
        return None


class NCMReadout(nn.Module):
    """Nearest class mean. Training-free; `fit` is incremental registration.

    Keeps one mean per class in the representation space it is given. Means are
    running averages, so the online case is exact (no need to store samples):
    `mu <- (n mu + n_batch mean) / (n + n_batch)`.
    """

    def __init__(self, dim: int, num_classes: int, scale: float = 10.0):
        super().__init__()
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.register_buffer("means", torch.zeros(num_classes, dim))
        self.register_buffer("counts", torch.zeros(num_classes))

    @torch.no_grad()
    def fit(
        self, z: torch.Tensor, y: torch.Tensor, seen_classes: list[int] | None = None
    ) -> dict:
        z = z.detach().float()
        y = y.detach().long()
        updated = 0
        for c in torch.unique(y).tolist():
            mask = y == c
            if not bool(mask.any()):
                continue
            batch_mean = z[mask].mean(dim=0)
            n = float(mask.sum())
            total = float(self.counts[c]) + n
            self.means[c] = (
                self.means[c] * float(self.counts[c]) + batch_mean * n
            ) / total
            self.counts[c] += n
            updated += 1
        return {
            "classes_updated": updated,
            "classes_seen": int((self.counts > 0).sum()),
        }

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        means = F.normalize(self.means, dim=-1)
        return self.scale * (F.normalize(z, dim=-1) @ means.t())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.predict(z)

    @property
    def seen_classes(self) -> list[int]:
        return torch.nonzero(self.counts > 0).flatten().tolist()


class CosineReadout(nn.Module):
    """`scale * <normalize(z), normalize(W)>`. The E0 reference readout.

    `freeze_rows` freezes the rows of classes already learned, which is the
    class-incremental convention E0 used (`adapter_r8` = 70.56 came from this
    readout plus per-task adapters).
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        scale: float = 10.0,
        steps: int = 200,
        lr: float = 1e-3,
        freeze_rows: bool = True,
        weight_decay: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.steps = int(steps)
        self.lr = float(lr)
        self.freeze_rows = bool(freeze_rows)
        self.weight_decay = float(weight_decay)
        self.W = nn.Parameter(torch.randn(num_classes, dim) / dim**0.5)
        # A bias is off by default because the E0 reference numbers were
        # produced without one. It matters in practice: without it the decision
        # surface is constrained in the angular sense and, on a symmetric
        # two-class problem, the two row directions can collapse onto each other
        # (measured: 52.5% without the bias, 83.3% with it, same budget). The
        # E0 reference worked because its problem had many classes.
        self.bias = nn.Parameter(torch.zeros(num_classes)) if bias else None
        self.register_buffer(
            "trainable_rows", torch.zeros(num_classes, dtype=torch.bool)
        )

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.scale * (F.normalize(z, dim=-1) @ F.normalize(self.W, dim=-1).t())
        return logits if self.bias is None else logits + self.bias

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.predict(z)

    def trainable_hook(self, active_classes: list[int]):
        """Install the readout's freezing policy and return the handle.

        A readout decides which of its rows may move in a given update; the loop
        must not decide that for it. S2 measured the cost of getting this wrong:
        re-opening every seen row collapses a linear readout to 7.69%
        (forgetting 92.35%) where the same readout with the policy reaches
        66.75%. The freeze is part of the readout's definition.
        """
        mask = torch.zeros(self.num_classes, dtype=torch.bool)
        mask[torch.tensor(list(active_classes), dtype=torch.long)] = True
        self.trainable_rows.copy_(mask)

        def _mask_rows(grad, mask=mask):
            if grad is None:
                return grad
            grad = grad.clone()
            grad[~mask.to(grad.device)] = 0.0
            return grad

        return self.W.register_hook(_mask_rows)

    def fit(
        self, z: torch.Tensor, y: torch.Tensor, seen_classes: list[int] | None = None
    ) -> dict:
        rows = seen_classes if seen_classes is not None else torch.unique(y).tolist()
        handle = self.trainable_hook(list(rows))
        opt = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        last = 0.0
        for _ in range(self.steps):
            opt.zero_grad()
            loss = F.cross_entropy(mask_unseen(self.predict(z), list(rows)), y)
            loss.backward()
            opt.step()
            last = float(loss.item())
        handle.remove()
        return {"loss": last, "trainable_rows": int(self.trainable_rows.sum())}


class LinearReadout(CosineReadout):
    """Plain linear layer (no cosine normalization). Same training contract."""

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.W)


class RidgeReadout(nn.Module):
    """Closed-form ridge regression on one-hot targets. No optimizer, no steps.

    This is the mentor's "start from linear regression" rung, made exact:
    `W = (Z^T Z + lambda I)^-1 Z^T Y`, computed in one shot. It is the cheapest
    non-trivial readout and therefore the fairest baseline for any claim that a
    trained head is needed.

    Incrementality is exact rather than approximate: the sufficient statistics
    `A = Z^T Z` and `B = Z^T Y` are accumulated per task, so a later task
    extends the solution without storing any sample. A ones column is appended
    so the intercept is solved jointly (no centering, no re-derivation).
    """

    def __init__(self, dim: int, num_classes: int, ridge: float = 1.0):
        super().__init__()
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.ridge = float(ridge)
        d1 = self.dim + 1
        self.register_buffer("A", self.ridge * torch.eye(d1))
        self.register_buffer("B", torch.zeros(d1, self.num_classes))
        self.register_buffer("W", torch.zeros(self.num_classes, d1))
        self.register_buffer("n_samples", torch.zeros(()))
        self.fitted = False

    def _augment(self, z: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(z.size(0), 1, device=z.device, dtype=z.dtype)
        return torch.cat([z, ones], dim=1)

    @torch.no_grad()
    def fit(
        self, z: torch.Tensor, y: torch.Tensor, seen_classes: list[int] | None = None
    ) -> dict:
        z = self._augment(z.detach().float())
        y = y.detach().long()
        rows = seen_classes if seen_classes is not None else torch.unique(y).tolist()
        targets = torch.zeros(z.size(0), self.num_classes, device=z.device)
        targets[torch.arange(z.size(0), device=z.device), y] = 1.0
        self.A += z.t() @ z
        self.B += z.t() @ targets
        self.n_samples += z.size(0)
        self.W.copy_(torch.linalg.solve(self.A, self.B).t())
        self.fitted = True
        return {
            "fitted_rows": len(rows),
            "n_samples": int(self.n_samples),
            "condition": float(torch.linalg.cond(self.A)),
        }

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(self._augment(z.float()), self.W)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.predict(z)


class LogisticReadout(LinearReadout):
    """Linear layer trained by cross-entropy: multinomial logistic regression.

    Registered separately from `linear` so the readout ladder in a config reads
    the same as the ladder in the paper, even though the two share an
    implementation (a linear layer under cross-entropy *is* logistic
    regression; the distinction is naming, not architecture).
    """


class MLPReadout(nn.Module):
    """A small MLP head: the top of the readout ladder before an expert bank."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        hidden: int = 256,
        steps: int = 200,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.steps = int(steps)
        self.lr = float(lr)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, num_classes)
        )

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.predict(z)

    def fit(
        self, z: torch.Tensor, y: torch.Tensor, seen_classes: list[int] | None = None
    ) -> dict:
        rows = seen_classes if seen_classes is not None else torch.unique(y).tolist()
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        last = 0.0
        for _ in range(self.steps):
            opt.zero_grad()
            loss = F.cross_entropy(mask_unseen(self.predict(z), list(rows)), y)
            loss.backward()
            opt.step()
            last = float(loss.item())
        return {"loss": last}


register_readout("ncm")(
    lambda dim, num_classes, scale=10.0: NCMReadout(dim, num_classes, scale=scale)
)
register_readout("cosine")(
    lambda dim, num_classes, **kw: CosineReadout(dim, num_classes, **kw)
)
register_readout("linear")(
    lambda dim, num_classes, **kw: LinearReadout(dim, num_classes, **kw)
)
register_readout("logistic")(
    lambda dim, num_classes, **kw: LogisticReadout(dim, num_classes, **kw)
)
register_readout("ridge")(
    lambda dim, num_classes, **kw: RidgeReadout(dim, num_classes, **kw)
)
register_readout("mlp")(
    lambda dim, num_classes, **kw: MLPReadout(dim, num_classes, **kw)
)
