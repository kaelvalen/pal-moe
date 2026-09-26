"""Consolidation policies: which data trains which (frozen) expert.

- `by_arrival`   one expert per written batch / task, in arrival order. This is the
                 Stage 1 behaviour (L3_per_task) and the **control**.
- `by_confusion` expert boundaries follow the router's confusion structure: spectral
                 clustering of the symmetrised class confusion matrix. **Experimental,
                 gated**: it may only run under `docs/P2_BOUND_PREREG.md`, and
                 `consolidate()` refuses it unless the caller names that document.
"""

from __future__ import annotations

import numpy as np
import torch

GATED = {"by_confusion": "docs/P2_BOUND_PREREG.md"}


class PolicyGateError(RuntimeError):
    pass


def check_gate(policy: str, prereg: str | None) -> None:
    need = GATED.get(policy)
    if need is not None and prereg != need:
        raise PolicyGateError(
            f"policy {policy!r} is experimental; pass prereg={need!r} to run it "
            "(and only after that pre-registration is committed)"
        )


def by_arrival(batch_classes: list[list[int]]) -> list[list[int]]:
    """Identity grouping: group i = the classes first introduced by batch i."""
    owned: set[int] = set()
    groups = []
    for classes in batch_classes:
        g = [c for c in classes if c not in owned]
        owned.update(g)
        groups.append(g)
    return groups


def confusion_matrix(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    idx = target.long() * num_classes + pred.long()
    return (
        torch.bincount(idx, minlength=num_classes**2)
        .reshape(num_classes, num_classes)
        .double()
    )


def by_confusion(
    confusion: torch.Tensor,
    n_groups: int,
    classes: list[int] | None = None,
    seed: int = 0,
    balanced: bool = True,
) -> list[list[int]]:
    """Spectral clustering of classes on the symmetrised, row-normalised confusion.

    Affinity `S = (P + P^T) / 2` with `P` the row-normalised confusion (diagonal
    removed), normalised Laplacian embedding on the `n_groups` smallest eigenvectors,
    then k-means (sklearn, fixed `random_state`). With `balanced=True` classes are
    assigned greedily to the nearest centroid with capacity `ceil(C / n_groups)`, so
    each expert gets a comparable share (the `by_arrival` control has equal-size
    tasks, and an unbalanced grouping would confound the comparison with capacity).
    """
    from sklearn.cluster import KMeans

    C = confusion.size(0)
    classes = list(range(C)) if classes is None else list(classes)
    M = (
        confusion[np.ix_(classes, classes)]
        if isinstance(confusion, np.ndarray)
        else confusion[classes][:, classes]
    )
    P = M.double().clone()
    P.fill_diagonal_(0.0)
    P = P / P.sum(1, keepdim=True).clamp(min=1e-12)
    S = 0.5 * (P + P.t())
    d = S.sum(1).clamp(min=1e-12)
    L = torch.eye(len(classes), dtype=S.dtype) - S / torch.sqrt(d[:, None] * d[None, :])
    _, vecs = torch.linalg.eigh(L)
    emb = vecs[:, :n_groups].numpy()
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    km = KMeans(n_clusters=n_groups, n_init=10, random_state=seed).fit(emb)
    if not balanced:
        labels = km.labels_
    else:
        cap = -(-len(classes) // n_groups)
        dist = ((emb[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
        labels = np.full(len(classes), -1)
        load = np.zeros(n_groups, dtype=int)
        for flat in np.argsort(dist, axis=None, kind="stable"):
            i, g = divmod(int(flat), n_groups)
            if labels[i] < 0 and load[g] < cap:
                labels[i], load[g] = g, load[g] + 1
    groups = [
        [classes[i] for i in range(len(classes)) if labels[i] == g]
        for g in range(n_groups)
    ]
    return [g for g in groups if g]


@torch.no_grad()
def cross_fitted_confusion(
    z: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    folds: int = 5,
    seed: int = 0,
    ridge: float = 1.0,
) -> torch.Tensor:
    """Class confusion of a float64 ridge, each sample predicted by a model that did
    not see it (`folds`-fold cross-fitting, folds from `torch.randperm(seed)`).
    Rows = true class, columns = predicted class. Uses only the data it is given."""
    from pal_moe.edit.stats import LinearStats, one_hot

    n = z.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    fold_of = torch.empty(n, dtype=torch.long)
    fold_of[perm] = torch.arange(n) % folds
    fold_of = fold_of.to(z.device)
    pred = torch.empty(n, dtype=torch.long, device=z.device)
    for f in range(folds):
        tr, te = fold_of != f, fold_of == f
        s = LinearStats(z.size(1), num_classes, ridge=ridge, device=z.device)
        s.add(s.contribution("fold", z[tr], one_hot(y[tr], num_classes)))
        pred[te] = s.predict(z[te]).argmax(-1)
    return confusion_matrix(pred.cpu(), y.cpu(), num_classes)


POLICIES = {
    "by_arrival": by_arrival,
    "by_confusion": by_confusion,
    "by_partition": None,  # explicit groups passed to consolidate()
}
