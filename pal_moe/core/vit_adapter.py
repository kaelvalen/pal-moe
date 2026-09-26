"""A frozen timm ViT with per-block parallel adapters (the EASE / AdaptFormer form).

Each transformer block gets a side branch on the MLP: `a = s * up(relu(down(x)))`
on the post-attention residual stream `x`, added to the MLP output; `down: d -> r`,
`up: r -> d` initialised to **zero**, so every adapter is the identity at init
(function-preserving, the S1 `RepresentationExpert` rule). `s = 0.1`, `r = 16`
follow EASE (CVPR 2024), which inserts "a side branch for the MLP" with projection
dim 16.

The base weights are frozen and hashed; only adapters (and whatever head the caller
trains) receive gradients. Several adapter sets can be held at once and one selected
per forward (`set_active`), which is what a routed expert bank over one backbone
needs. The frozen [CLS] (no adapter active) is the fixed address.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .hashing import module_digest

DEFAULT_BACKBONE = "vit_base_patch16_224.augreg_in21k"


class Adapter(nn.Module):
    def __init__(self, dim: int, rank: int = 16, scale: float = 0.1):
        super().__init__()
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        self.scale = scale
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.scale * self.up(torch.relu(self.down(x)))


class AdaptedViT(nn.Module):
    def __init__(
        self, name: str = DEFAULT_BACKBONE, pretrained: bool = True, rank: int = 16
    ):
        super().__init__()
        import timm

        self.vit = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.name, self.rank = name, rank
        self.dim = self.vit.num_features
        self.base_hash = module_digest(self.vit)
        self.adapters = nn.ModuleDict()  # expert id -> ModuleList over blocks
        self.active: str | None = None
        for blk in self.vit.blocks:
            blk.forward = self._block_forward(blk)

    def _block_forward(self, blk):
        idx = list(self.vit.blocks).index(blk)

        def forward(x, *args, **kwargs):
            x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
            out = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
            if self.active is not None:
                out = out + self.adapters[self.active][idx](x)
            return out

        return forward

    def add_expert(self, expert_id: str) -> nn.ModuleList:
        mods = nn.ModuleList(Adapter(self.dim, self.rank) for _ in self.vit.blocks)
        dev = next(self.vit.parameters()).device
        self.adapters[expert_id] = mods.to(dev)
        return self.adapters[expert_id]

    def freeze_expert(self, expert_id: str) -> None:
        for p in self.adapters[expert_id].parameters():
            p.requires_grad_(False)

    def set_active(self, expert_id: str | None) -> None:
        self.active = expert_id

    def forward(self, x: torch.Tensor, expert_id: str | None = None) -> torch.Tensor:
        """[CLS] features (pre-logits) with `expert_id`'s adapters (None = frozen)."""
        prev, self.active = self.active, expert_id
        try:
            return self.vit(x)
        finally:
            self.active = prev

    def data_config(self) -> dict:
        import timm

        return timm.data.resolve_data_config({}, model=self.vit)
