"""
Router implementations for the S1 contract: `Z -> distribution over experts`.

Kept separate from both expert and readout on purpose. The measured reason
(E0): candidate recall@3 is 88.6% while oracle routing is 97.64%, so *selection*
is worth 27 points and it is a different object from adaptation or readout. A
router that is hidden inside an expert or a readout cannot be measured
separately, which is why v1 could not tell "the experts are bad" from "the
selection is bad".

Registered names:

    prototype      nearest class mean in `z` -> that class's expert
    legacy_linear  the v1 `DynamicRouter` behind the same interface
    legacy_distance
    legacy_attention
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pal_moe.legacy.models.router import AttentionRouter, DistanceRouter, DynamicRouter
from .registry import register_router

__all__ = ["PrototypeRouter", "LegacyRouter"]


class PrototypeRouter(nn.Module):
    """Nearest-class-mean router: the candidate generator E0 measured.

    Holds one mean per class and a class -> expert map. Routing is the softmax
    over the best per-expert similarity, so `route()` is a proper distribution
    and `top_k()` exposes the candidate set that recall@K is defined on.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        num_experts: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.num_experts = int(num_experts)
        self.temperature = float(temperature)
        self.register_buffer("means", torch.zeros(num_classes, dim))
        self.register_buffer("counts", torch.zeros(num_classes))
        self.register_buffer(
            "class_expert", torch.full((num_classes,), -1, dtype=torch.long)
        )

    @torch.no_grad()
    def register_class(self, class_id: int, expert_id: int, z: torch.Tensor) -> None:
        """Set (or update) a class's prototype and which expert owns it."""
        z = z.detach().float().view(-1)
        self.means[class_id] = z
        self.counts[class_id] += 1
        self.class_expert[class_id] = int(expert_id)

    @torch.no_grad()
    def register_from_batch(
        self, z: torch.Tensor, y: torch.Tensor, expert_id: int
    ) -> None:
        for c in torch.unique(y).tolist():
            self.register_class(int(c), expert_id, z[y == c].mean(dim=0))

    def expert_scores(self, z: torch.Tensor) -> torch.Tensor:
        """[B, num_experts] best cosine similarity per expert."""
        seen = torch.nonzero(self.counts > 0).flatten()
        if seen.numel() == 0:
            return torch.full(
                (z.size(0), self.num_experts), -2.0, device=z.device, dtype=z.dtype
            )
        sim = F.normalize(z, dim=-1) @ F.normalize(self.means[seen], dim=-1).t()
        owners = self.class_expert[seen].clamp(min=0)
        scores = torch.full(
            (z.size(0), self.num_experts), -2.0, device=z.device, dtype=z.dtype
        )
        scores.scatter_reduce_(
            1, owners.unsqueeze(0).expand(z.size(0), -1), sim, reduce="amax"
        )
        return scores

    def route(self, z: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.expert_scores(z) / self.temperature, dim=-1)

    def top_k(self, z: torch.Tensor, k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Candidate expert ids and their scores, best first."""
        scores = self.expert_scores(z)
        values, indices = scores.topk(min(k, self.num_experts), dim=-1)
        return indices, values

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.route(z)


class LegacyRouter(nn.Module):
    """`Router` view of a v1 router. Adds no parameters and changes no numerics.

    `top_k` is derived from the same dense distribution the v1 router returns,
    so the candidate set is well defined for the v1 path too (it was not
    reported before S1).
    """

    def __init__(self, router: nn.Module):
        super().__init__()
        self.router = router
        self.num_experts = int(router.num_experts)
        self.input_dim = int(router.input_dim)

    def route(self, z: torch.Tensor) -> torch.Tensor:
        return self.router.get_full_distribution(z)

    def top_k(self, z: torch.Tensor, k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.route(z)
        values, indices = scores.topk(min(k, self.num_experts), dim=-1)
        return indices, values

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.route(z)


register_router("prototype")(
    lambda dim, num_classes, num_experts, temperature=1.0: PrototypeRouter(
        dim, num_classes, num_experts, temperature=temperature
    )
)
register_router("legacy_linear")(lambda **kw: LegacyRouter(DynamicRouter(**kw)))
register_router("legacy_distance")(lambda **kw: LegacyRouter(DistanceRouter(**kw)))
register_router("legacy_attention")(lambda **kw: LegacyRouter(AttentionRouter(**kw)))


def build_legacy_router(router_type: str, input_dim: int, top_k: int, num_experts: int):
    """Map the v1 `--router_type` values onto the S1 interface."""
    mapping = {
        "dynamic": "legacy_linear",
        "distance": "legacy_distance",
        "attention": "legacy_attention",
    }
    if router_type not in mapping:
        raise KeyError(f"unknown router_type {router_type!r}; use {sorted(mapping)}")
    return build_router(
        mapping[router_type],
        input_dim=input_dim,
        top_k=top_k,
        num_experts=num_experts,
    )


from .registry import build_router  # noqa: E402  (bottom: avoids import cycle)
