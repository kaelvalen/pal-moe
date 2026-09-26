"""
Evaluation-time classifier heads.

Class-incremental models develop a recency bias: the newest classes have larger
logits because they were trained last. Two cheap, well-established remedies
that need no training:

- `NCMHead`: nearest-class-mean over the stored latent exemplars (the iCaRL /
  RanPAC style read-out), which sidesteps the biased linear classifier;
- `BiasCorrectionHead`: subtracts a fitted per-class logit prior (estimated on
  the stored exemplars) from the model's logits.

Both wrap a PAL-MoE model and expose the same `forward(x) -> logits` contract,
so they plug straight into `ContinualEvaluator`.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["NCMHead", "BiasCorrectionHead"]


class NCMHead(nn.Module):
    """Nearest-class-mean read-out over prototype-memory exemplars."""

    def __init__(
        self,
        model: nn.Module,
        prototype_memory: Any,
        num_classes: int,
        temperature: float = 10.0,
    ):
        super().__init__()
        self.model = model
        self.prototype_memory = prototype_memory
        self.num_classes = num_classes
        self.temperature = temperature
        self.register_buffer("class_means", torch.zeros(num_classes, 0))
        self.refit()

    @torch.no_grad()
    def refit(self) -> bool:
        """Recomputes class means from all labeled stored exemplars."""
        batch = self.prototype_memory.get_exemplar_batch(
            next(self.model.parameters()).device
        )
        if batch is None:
            return False
        feats, labels = batch
        feat_dim = feats.size(1)
        sums = torch.zeros(self.num_classes, feat_dim, device=feats.device)
        counts = torch.zeros(self.num_classes, device=feats.device)
        sums.index_add_(0, labels, feats)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
        means = sums / counts.clamp(min=1).unsqueeze(1)
        means[counts == 0] = 0.0
        self.class_means = means.cpu()
        self._valid_classes = counts > 0
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.class_means.numel() == 0 or self.class_means.size(0) == 0:
            return self.model(x)
        h = (
            self.model.get_routing_features(x)
            if hasattr(self.model, "get_routing_features")
            else self.model(x)
        )
        means = self.class_means.to(h.device)
        h_n = F.normalize(h, p=2, dim=1)
        m_n = F.normalize(means, p=2, dim=1)
        logits = h_n @ m_n.t() * self.temperature
        # Unseen classes must never win.
        if hasattr(self, "_valid_classes"):
            logits = logits.masked_fill(
                ~self._valid_classes.to(h.device).unsqueeze(0), -1e9
            )
        return logits


class BiasCorrectionHead(nn.Module):
    """Subtracts a fitted per-class logit prior (anti-recency-bias read-out)."""

    def __init__(
        self,
        model: nn.Module,
        prototype_memory: Any,
        num_classes: int,
        strength: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.prototype_memory = prototype_memory
        self.num_classes = num_classes
        self.strength = strength
        self.register_buffer("prior", torch.zeros(num_classes))
        self.refit()

    @torch.no_grad()
    def refit(self) -> bool:
        """Mean logit of every output class over the stored exemplars."""
        batch = self.prototype_memory.get_exemplar_batch(
            next(self.model.parameters()).device
        )
        if batch is None:
            self.prior = torch.zeros(self.num_classes)
            return False
        feats, labels = batch
        logits = self.model(latent_h=feats)  # [M, C]
        self.prior = logits.mean(dim=0).cpu()
        counts = torch.zeros(self.num_classes, device=labels.device)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
        self._valid_classes = counts > 0
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        if self.prior.numel() == 0:
            return logits
        prior = self.prior.to(logits.device)
        corrected = logits - self.strength * prior.unsqueeze(0)
        if hasattr(self, "_valid_classes"):
            corrected = corrected.masked_fill(
                ~self._valid_classes.to(logits.device).unsqueeze(0), -1e9
            )
        return corrected
