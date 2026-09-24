"""
Expert implementations for the S1 contract: `Z -> Z'`.

The rule that makes an expert an expert here: **it must be identity at
initialization**. That is what makes adding one function-preserving (E0 uses it
to keep the L2/L3 comparison clean) and what separates adaptation from
classification. A module that maps `Z -> Y` is a `ClassificationExpert` and
belongs to the legacy registry, because it cannot be composed with a readout.

Registered names:

    identity            no parameters, always exactly the identity
    residual_adapter    `z + U gelu(V z)`, U zero-init, rank `r`
    mlp                 a small residual MLP, output layer zero-init
    legacy_mlp          the v1 `MLPExpert` (`Z -> Y`) as a ClassificationExpert
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.expert import MLPExpert
from .registry import (
    register_classification_expert,
    register_expert,
)

__all__ = [
    "IdentityExpert",
    "ResidualAdapter",
    "ResidualMLPExpert",
    "LegacyClassificationExpert",
]


class IdentityExpert(nn.Module):
    """`z -> z`. The zero-capacity expert: the reference a real expert must beat."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def transform(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.transform(z)


class ResidualAdapter(nn.Module):
    """`A(z) = U @ gelu(V @ z)`, `U` zero-init so `A == 0` exactly.

    Rank is the capacity knob; E0 measured that rank 8 already saturates the
    per-task case (r8 70.56, r32 70.62, r64 70.60), so the default is 8 and
    larger ranks are for the capacity-budget axis, not for tuning.
    """

    def __init__(self, dim: int, rank: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        if rank > 0:
            self.V = nn.Parameter(torch.empty(rank, dim))
            self.U = nn.Parameter(torch.zeros(dim, rank))
            nn.init.kaiming_uniform_(self.V, a=5**0.5)
        else:
            self.register_parameter("V", None)
            self.register_parameter("U", None)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def transform(self, z: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return z
        hidden = self.dropout(F.gelu(F.linear(z, self.V)))
        return z + hidden @ self.U.t()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.transform(z)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ResidualMLPExpert(nn.Module):
    """A wider residual expert: `z + W2 gelu(W1 z)`, `W2` zero-init.

    The middle rung between the rank-limited adapter and a full MLP transform:
    it is still `Z -> Z` and still the identity at init, so it composes with any
    readout.
    """

    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def transform(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.fc2(self.dropout(F.gelu(self.fc1(z))))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.transform(z)


class LegacyClassificationExpert(nn.Module):
    """`ClassificationExpert` view of the v1 `MLPExpert` (`Z -> Y`).

    This class exists to make the abstraction error visible rather than to fix
    it: v1 fuses adaptation and readout in one module, which is exactly why the
    L2 (capacity) and L3 (isolation) effects could not be separated before S1.
    v1 code keeps using `MLPExpert` directly; nothing is migrated here, and the
    wrapper adds no parameters and changes no numerics.
    """

    def __init__(self, expert: MLPExpert):
        super().__init__()
        self.expert = expert
        self.num_classes = int(expert.num_classes)
        self.input_dim = int(expert.input_dim)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.expert(z, track_usage=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.classify(z)


register_expert("identity")(lambda dim: IdentityExpert(dim))
register_expert("residual_adapter")(
    lambda dim, rank=8, dropout=0.0: ResidualAdapter(dim, rank=rank, dropout=dropout)
)
register_expert("mlp")(
    lambda dim, hidden=256, dropout=0.0: ResidualMLPExpert(
        dim, hidden=hidden, dropout=dropout
    )
)
register_classification_expert("legacy_mlp")(
    lambda input_dim, hidden_dim=256, num_classes=10, **kw: LegacyClassificationExpert(
        MLPExpert(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            **kw,
        )
    )
)


def expert_param_count(expert: nn.Module) -> int:
    return sum(p.numel() for p in expert.parameters())
