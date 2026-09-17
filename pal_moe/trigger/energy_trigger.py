"""
Energy-based novelty trigger.

The composite trigger (`QuantitativeTrigger`) scores a batch with task loss,
router entropy, prototype distance and confidence. For task-free / production
streams a purely unsupervised novelty signal is often preferable: the energy of
a classifier, E(x) = logsumexp(logits), is low for in-distribution data and
rises on out-of-distribution inputs.

`EnergyTrigger` maintains an EMA reference distribution of energies on data the
model has seen (updated whenever `evaluate` is called on routine batches) and
triggers expansion when the current batch's energy is `threshold` standard
deviations above the reference.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .expert_trigger import TriggerEvaluationResult

__all__ = ["EnergyTrigger"]


def energy(logits: torch.Tensor) -> torch.Tensor:
    """E(x) = logsumexp(logits) per sample."""
    return torch.logsumexp(logits, dim=-1)


@dataclass
class _RunningStats:
    mean: float = 0.0
    var: float = 1.0
    count: int = 0

    def update(self, values: torch.Tensor, momentum: float = 0.1) -> None:
        batch_mean = float(values.mean().item())
        batch_var = (
            float(values.var(unbiased=False).item()) if values.numel() > 1 else 0.0
        )
        if self.count == 0:
            self.mean, self.var = batch_mean, max(batch_var, 1e-6)
        else:
            self.mean = (1 - momentum) * self.mean + momentum * batch_mean
            self.var = (1 - momentum) * self.var + momentum * max(batch_var, 1e-6)
        self.count += 1


class EnergyTrigger:
    """
    Unsupervised expansion trigger based on the energy z-score.

    `update_reference=False` allows scoring a batch without letting it move the
    reference distribution (e.g. during an explicit calibration pass).
    """

    def __init__(
        self,
        threshold: float = 3.0,
        momentum: float = 0.1,
        warmup_batches: int = 4,
    ):
        self.threshold = threshold
        self.momentum = momentum
        self.warmup_batches = warmup_batches
        self.stats = _RunningStats()

    @torch.no_grad()
    def evaluate(
        self,
        model: Any,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        prototype_memory: Optional[Any] = None,
    ) -> TriggerEvaluationResult:
        model.eval()
        h = (
            model.get_routing_features(x)
            if hasattr(model, "get_routing_features")
            else x
        )
        # Batch energy = mean softmax-expert energy over the routed expert.
        weights, topk_idx, _ = model.router(h)
        energies = []
        for expert_idx in torch.unique(topk_idx).tolist():
            mask = (topk_idx == expert_idx).any(dim=-1)
            if mask.any():
                logits = model.experts[expert_idx](h[mask], track_usage=False)
                energies.append(energy(logits).mean())
        batch_energy = float(torch.stack(energies).mean().item()) if energies else 0.0

        # Parent selection: router's most frequent expert on the batch.
        counts = torch.bincount(topk_idx.flatten(), minlength=model.num_experts)
        best_parent = int(counts.argmax().item())

        ref_mean, ref_std = self.stats.mean, max(self.stats.var**0.5, 1e-6)
        z_score = (batch_energy - ref_mean) / ref_std

        warmed_up = self.stats.count >= self.warmup_batches
        should_trigger = bool(warmed_up and z_score > self.threshold)
        if not should_trigger:
            self.stats.update(torch.tensor([batch_energy]), momentum=self.momentum)

        return TriggerEvaluationResult(
            should_trigger=should_trigger,
            composite_score=float(z_score),
            loss_best_expert=0.0,
            router_entropy=0.0,
            proto_distance=0.0,
            max_confidence=0.0,
            best_parent_expert_idx=best_parent,
        )
