"""`prototype`: the E0 / AC3 rule - best cosine to a stored class mean, per expert.

Wraps `pal_moe.arch.routers.PrototypeRouter` (buffers only). When the expert bank
comes from the Stage 1 ladder, the bank's own router object is wrapped as-is, so
the routed ids are bitwise the ones S11 / E-TID2 measured.
"""

from __future__ import annotations

import torch

from pal_moe.arch.routers import PrototypeRouter


class PrototypeTaskRouter:
    name = "prototype"

    def __init__(self, router: PrototypeRouter):
        self.router = router

    @classmethod
    def empty(
        cls, dim: int, num_classes: int, num_experts: int, device="cpu"
    ) -> PrototypeTaskRouter:
        return cls(
            PrototypeRouter(
                dim=dim, num_classes=num_classes, num_experts=num_experts
            ).to(device)
        )

    @torch.no_grad()
    def register(self, z: torch.Tensor, y: torch.Tensor, expert_id: int) -> None:
        self.router.register_from_batch(z, y, expert_id)

    @torch.no_grad()
    def scores(self, z: torch.Tensor) -> torch.Tensor:
        return self.router.expert_scores(z)

    @torch.no_grad()
    def top_k(self, z: torch.Tensor, k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        return self.router.top_k(z, k=k)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.router.parameters())
