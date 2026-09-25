"""
Representation-geometry diagnostics.

Continual-learning heuristics (prototype memory, OOD boundaries, router
anchoring) only work when the latent space actually separates classes/tasks.
These cheap metrics make that assumption measurable instead of implicit:

- nearest-other margin: for each sample, distance to the nearest different-class
  sample minus distance to the nearest same-class sample (positive = the point
  sits inside its own class region);
- silhouette: standard silhouette score on a sample of the features;
- class-mean separation: minimum pairwise distance between class means divided
  by the mean pairwise distance (a scale-free clustering quality proxy).

All functions are torch-only and safe to call on CUDA tensors.
"""

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "nearest_other_margin",
    "silhouette_score",
    "geometry_report",
]


@torch.no_grad()
def nearest_other_margin(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sample (d_other - d_same); positive means correctly clustered."""
    if features.size(0) < 2:
        return torch.zeros(features.size(0), device=features.device)
    dists = torch.cdist(features, features)
    dists.fill_diagonal_(float("inf"))
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    d_same = dists.masked_fill(~same, float("inf")).min(dim=1).values
    d_other = dists.masked_fill(same, float("inf")).min(dim=1).values
    # Samples that are the only member of their class have no same-class
    # neighbour: fall back to 0 margin instead of inf.
    d_same = torch.nan_to_num(d_same, posinf=0.0)
    return d_other - d_same


@torch.no_grad()
def silhouette_score(
    features: torch.Tensor,
    labels: torch.Tensor,
    max_samples: int = 2000,
    seed: int = 0,
) -> float:
    """Silhouette score in [-1, 1] (subsampled for large inputs)."""
    n = features.size(0)
    if n < 3:
        return 0.0
    if n > max_samples:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(n, generator=gen)[:max_samples]
        features = features[idx]
        labels = labels[idx]
    dists = torch.cdist(features, features)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    same = same & ~eye
    other = ~same & ~eye

    def _mean_or_zero(masked):
        counts = masked.sum(dim=1).clamp(min=1)
        return (dists * masked).sum(dim=1) / counts

    a = _mean_or_zero(same)
    b = _mean_or_zero(other)
    denom = torch.maximum(a, b)
    valid = denom > 0
    if not bool(valid.any()):
        return 0.0
    s = ((b - a) / denom)[valid]
    return float(s.mean().item())


@torch.no_grad()
def geometry_report(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_means: Optional[torch.Tensor] = None,
    max_samples: int = 2000,
) -> dict[str, float]:
    """
    Bundles the per-space diagnostics.

    `class_means` can be passed when the caller already has class centroids
    (e.g. prototype memory); otherwise they are derived from the features.
    """
    features = F.normalize(features.float(), p=2, dim=1)
    margin = nearest_other_margin(features, labels)
    report = {
        "n_samples": int(features.size(0)),
        "num_classes": int(labels.unique().numel()),
        "margin_mean": float(margin.mean().item()),
        "margin_std": float(margin.std().item()),
        "margin_positive_frac": float((margin > 0).float().mean().item()),
        "silhouette": silhouette_score(features, labels, max_samples=max_samples),
    }
    if class_means is None:
        unique = labels.unique()
        class_means = torch.stack(
            [features[labels == c].mean(dim=0) for c in unique], dim=0
        )
    else:
        class_means = F.normalize(class_means.float(), p=2, dim=1)
    if class_means.size(0) >= 2:
        mean_dists = torch.cdist(class_means, class_means)
        eye = torch.eye(class_means.size(0), dtype=torch.bool, device=mean_dists.device)
        off_diag = mean_dists[~eye]
        report["class_mean_min_dist"] = float(off_diag.min().item())
        report["class_mean_mean_dist"] = float(off_diag.mean().item())
        report["class_mean_separation_ratio"] = float(
            (off_diag.min() / (off_diag.mean() + 1e-9)).item()
        )
    return report
