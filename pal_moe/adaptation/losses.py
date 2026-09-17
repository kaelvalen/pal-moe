"""
Loss composition helpers for the continual trainer.

`UncertaintyWeighter` implements Kendall et al.'s homoscedastic uncertainty
weighting: every core loss term gets a learnable log-variance s_i and the total
is `sum_i exp(-s_i) * L_i + s_i`. The model then decides how much to trust the
task loss versus the stability/OOD terms instead of a hand-tuned fixed ratio.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

__all__ = ["UncertaintyWeighter"]


class UncertaintyWeighter(nn.Module):
    """Learnable multi-task weighting for a fixed set of loss names."""

    def __init__(self, names: Iterable[str] = ("task", "router", "expert", "ood")):
        super().__init__()
        self.names = tuple(names)
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in self.names}
        )

    def combine(
        self, losses: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns (total, diagnostics) where every named loss is precision-weighted
        and unlisted losses pass through unchanged.
        """
        total = torch.zeros((), device=next(iter(losses.values())).device)
        diagnostics: dict[str, float] = {}
        for name, loss in losses.items():
            if name in self.log_vars:
                log_var = self.log_vars[name]
                weighted = torch.exp(-log_var) * loss + log_var
                total = total + weighted
                diagnostics[name] = float(torch.exp(-log_var).detach().item())
            else:
                total = total + loss
        return total, diagnostics
