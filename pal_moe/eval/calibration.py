"""
Confidence calibration utilities.

Temperature scaling fits one scalar per expert by minimising the negative
log-likelihood of that expert's predictions on the data it is routed. Under
top-1 routing the argmax is temperature-invariant, but calibration sharpens
confidence, improves top-k>1 mixtures and makes the gate/OOD thresholds
comparable across experts.
"""

from typing import Any, Optional

import torch
import torch.nn.functional as F

__all__ = ["fit_temperature", "calibrate_expert_temperatures"]


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    grid: Optional[torch.Tensor] = None,
) -> float:
    """Grid-search temperature T minimising CE(logits / T, labels)."""
    if logits.size(0) == 0 or labels.numel() == 0:
        return 1.0
    if grid is None:
        grid = torch.linspace(0.25, 4.0, 31)
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        nll = float(F.cross_entropy(logits / max(float(t), 1e-3), labels).item())
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


@torch.no_grad()
def calibrate_expert_temperatures(
    model: Any,
    loader: Any,
    max_batches: int = 8,
    min_samples: int = 10,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """
    Fits one temperature per expert on the samples the router sends to it.

    Returns {"temperatures": [T_0, ..., T_{N-1}], "samples": [...], "nll_before",
    "nll_after"} so callers can log the improvement.
    """
    device = device or next(model.parameters()).device
    num_experts = model.num_experts
    logits_per_expert: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]
    labels_per_expert: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]

    was_training = model.training
    model.eval()
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        h = model.get_routing_features(x)
        _, topk_idx, _ = model.router(h)
        for e_idx in range(num_experts):
            mask = (topk_idx == e_idx).any(dim=-1)
            if mask.any():
                logits_per_expert[e_idx].append(model.experts[e_idx](h[mask]))
                labels_per_expert[e_idx].append(y[mask])
    if was_training:
        model.train()

    temperatures = []
    samples = []
    nll_before = nll_after = 0.0
    for e_idx in range(num_experts):
        if not logits_per_expert[e_idx]:
            temperatures.append(1.0)
            samples.append(0)
            continue
        logits = torch.cat(logits_per_expert[e_idx], dim=0)
        labels = torch.cat(labels_per_expert[e_idx], dim=0)
        samples.append(int(logits.size(0)))
        if logits.size(0) < min_samples:
            temperatures.append(1.0)
            continue
        nll_before += float(F.cross_entropy(logits, labels).item())
        t = fit_temperature(logits, labels)
        nll_after += float(F.cross_entropy(logits / t, labels).item())
        temperatures.append(t)
    return {
        "temperatures": temperatures,
        "samples": samples,
        "nll_before": nll_before,
        "nll_after": nll_after,
    }
