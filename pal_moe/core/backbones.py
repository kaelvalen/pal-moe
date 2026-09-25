"""Frozen backbones as key producers.

The v3 contract: the backbone is immutable and content-hashed, and whatever it
emits at the chosen layer *is* the address. Nothing downstream may train it.
"""

from __future__ import annotations

import torch

from .hashing import digest, file_digest, module_digest


class FrozenFeatureBackbone:
    """Identity over pre-extracted features (the ViT-B/16 feature cache).

    `base_hash` names the backbone: the cache file's sha256 when a path is given,
    otherwise a caller-supplied string. `encode` is the identity, so every key is
    byte-identical to the cached feature - the same "fixed preprocessing" the S2
    ladder relies on.
    """

    def __init__(
        self, dim: int, base_hash: str | None = None, cache_path: str | None = None
    ):
        self.dim = int(dim)
        if cache_path is not None:
            base_hash = file_digest(cache_path)
        self.base_hash = base_hash or digest("frozen-features", self.dim)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def parameter_count(self) -> int:
        return 0


class FrozenModuleBackbone:
    """Any `nn.Module` encoder, frozen and hashed (torchvision, legacy encoders)."""

    def __init__(self, module: torch.nn.Module, dim: int):
        self.module = module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.dim = int(dim)
        self.base_hash = module_digest(self.module)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def parameter_count(self) -> int:
        return 0  # frozen: none trainable
