"""
Backbone implementations for the S1 contract: `X -> Z`.

Every backbone here wraps an existing module; none of them changes what the v1
encoder computes. The point is that `pal_moe` code can hold a `Backbone` and
never learn whether it is an MLP, a ResNet, a ViT or a cached feature matrix.

Registered names:

    mlp, conv, resnet18, resnet34, resnet50, vit_b_16, vit_b_32, vit_l_16
        torchvision / repo backbones, optional ImageNet weights (see
        `SharedEncoder`)
    cached
        an identity backbone over precomputed features (the feature-cache path)
    random
        a frozen random projection: the "random representation" condition that
        rule 1 of the Stage 1 brief requires
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.feature_cache import CachedFeatureEncoder
from pal_moe.legacy.models.encoder import SharedEncoder
from .registry import register_backbone

__all__ = [
    "SharedEncoderBackbone",
    "CachedBackbone",
    "ProjectedBackbone",
    "RandomProjectionBackbone",
    "RESNET_ARCHES",
    "VIT_ARCHES",
]


class SharedEncoderBackbone(nn.Module):
    """`Backbone` view of the repo's `SharedEncoder`.

    `encode` delegates to the encoder's forward; nothing is copied, so freezing,
    eval-mode handling and the feature cache keep working exactly as before.
    """

    def __init__(self, encoder: SharedEncoder):
        super().__init__()
        self.encoder = encoder
        self.output_dim = int(encoder.output_dim)
        self.arch = getattr(encoder, "arch", "unknown")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # convenience only
        return self.encode(x)

    def freeze(self) -> None:
        self.encoder.freeze()

    def unfreeze(self) -> None:
        self.encoder.unfreeze()


class CachedBackbone(nn.Module):
    """Identity `Backbone` over precomputed features (the feature-cache path).

    Mathematically identical to a frozen encoder: the cached tensors already are
    `z`. Kept as a separate class rather than reusing `CachedFeatureEncoder`
    directly so that the contract test can tell the two paths apart.
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = int(output_dim)
        self.arch = "cached"
        self._device_probe = nn.Parameter(torch.zeros(()), requires_grad=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def freeze(self) -> None:
        return None

    def unfreeze(self) -> None:
        return None


class ProjectedBackbone(nn.Module):
    """Frozen linear projection of a backbone's output to a canonical latent dim.

    Why this exists (S3): different backbones have different native widths (MLP
    256, ResNet 512, ViT 768, raw pixels 3072). Sweeping the backbone without a
    common `d0` would change the readout's parameter count, the prototype store
    size and the effective capacity at the same time as the backbone - four
    variables instead of one.

    The projection is frozen and seeded, so no trainable model enters the
    pipeline (a learned projection would be a new model, not a control). When
    `out_dim <= base.output_dim` the matrix is semi-orthogonal, which is
    norm-preserving and keeps the geometry comparable; when it is larger the
    projection is rank-deficient by construction and that is documented rather
    than hidden.
    """

    def __init__(
        self, base: nn.Module, out_dim: int, seed: int = 0, orthogonal: bool = True
    ):
        super().__init__()
        self.base = base
        self.output_dim = int(out_dim)
        self.base_dim = int(base.output_dim)
        self.arch = f"projected:{getattr(base, 'arch', 'unknown')}"
        generator = torch.Generator().manual_seed(int(seed))
        weight = torch.randn(self.base_dim, self.output_dim, generator=generator)
        if orthogonal and self.output_dim <= self.base_dim:
            # Semi-orthogonal: W^T W = I, so the projection preserves norms.
            q, _ = torch.linalg.qr(weight)
            weight = q[:, : self.output_dim]
        else:
            weight = weight / self.base_dim**0.5
        self.register_buffer("proj", weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.encode(x) @ self.proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def freeze(self) -> None:
        self.base.freeze()

    def unfreeze(self) -> None:
        return None


class RandomProjectionBackbone(nn.Module):
    """A frozen random projection: the "random representation" condition.

    Rule 1 of the Stage 1 brief asks for the same method to be tested across
    representation families, including a random one. A random projection is the
    cheapest honest version of that: no pretraining, no semantics, but the same
    `X -> Z` interface, so the rest of the pipeline is unchanged. Seeded, so the
    condition is reproducible and the projection is shared across runs with the
    same seed.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        seed: int = 0,
        nonlinear: bool = True,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.arch = "random"
        self.nonlinear = bool(nonlinear)
        generator = torch.Generator().manual_seed(int(seed))
        weight = torch.randn(output_dim, input_dim, generator=generator)
        self.register_buffer("weight", weight / input_dim**0.5)
        self.register_buffer("bias", torch.zeros(output_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = x.flatten(start_dim=1) @ self.weight.t() + self.bias
        return torch.relu(z) if self.nonlinear else z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        return None  # a random projection is frozen by construction


RESNET_ARCHES = SharedEncoder.RESNET_ARCHES
VIT_ARCHES = SharedEncoder.VIT_ARCHES


def _shared_encoder(
    input_dim: int,
    output_dim: int,
    arch: str,
    hidden_dims=None,
    conv_channels=(32, 64, 128),
    backbone_weights: str = "none",
    input_mean=None,
    input_std=None,
) -> SharedEncoderBackbone:
    """Wrap `SharedEncoder`, forwarding the dataset statistics it needs.

    `input_mean`/`input_std` are not cosmetic: the ViT path undoes the dataset
    normalisation before applying ImageNet statistics, and without them it
    double-normalises. S3 measured the effect: the ViT features had norm 7.8
    instead of 18.5 and every ladder rung collapsed (L3 28.9% vs the reference
    70.6%). Any backbone factory must be able to express them.
    """
    return SharedEncoderBackbone(
        SharedEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            arch=arch,
            conv_channels=tuple(conv_channels),
            backbone_weights=backbone_weights,
            input_mean=input_mean,
            input_std=input_std,
        )
    )


def _register_shared_encoder_archs() -> None:
    for arch in ("mlp", *SharedEncoder.RESNET_ARCHES, *SharedEncoder.VIT_ARCHES):
        if arch == "mlp":
            register_backbone("mlp")(
                lambda input_dim, output_dim=128, hidden_dims=(
                    256,
                    128,
                ), **kw: _shared_encoder(
                    input_dim, output_dim, "mlp", hidden_dims=hidden_dims, **kw
                )
            )
        else:
            register_backbone(arch)(
                lambda input_dim, output_dim=128, arch=arch, **kw: _shared_encoder(
                    input_dim, output_dim, arch, **kw
                )
            )
    register_backbone("conv")(
        lambda input_dim, output_dim=128, conv_channels=(
            32,
            64,
            128,
        ), **kw: _shared_encoder(
            input_dim, output_dim, "conv", conv_channels=conv_channels, **kw
        )
    )
    register_backbone("cached")(
        lambda output_dim: CachedBackbone(output_dim=output_dim)
    )
    register_backbone("random")(
        lambda input_dim, output_dim=256, seed=0, nonlinear=True: RandomProjectionBackbone(
            input_dim=input_dim,
            output_dim=output_dim,
            seed=seed,
            nonlinear=nonlinear,
        )
    )


# Native torchvision widths. The projection must be fed the backbone's OWN
# features, not a width the factory invented: S3 measured what happens
# otherwise. Asking for output_dim=256 from a ViT-B/16 (native 768) inserts a
# 768->256 BatchNorm+ReLU head and then projects back up, and the CIFAR-100
# features collapse (norm 4.3 instead of 18.4, NCM 28% instead of 70%).
NATIVE_WIDTHS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "vit_b_16": 768,
    "vit_b_32": 768,
    "vit_l_16": 1024,
}


def build_projected_backbone(
    arch: str,
    input_dim: int,
    latent_dim: int,
    rep_seed: int = 0,
    hidden_dims=(256, 128),
    conv_channels=(32, 64, 128),
    backbone_weights: str = "none",
    input_mean=None,
    input_std=None,
) -> nn.Module:
    """Build a backbone and freeze a projection to a common latent dimension.

    This is the S3 control: `latent_dim` is held fixed across the sweep, so the
    readout's parameter count and the prototype store stay constant while the
    backbone changes.

    Construction rule: ask the encoder for the backbone's **native** width (for
    torchvision arches), or for `latent_dim` directly when the width is a free
    choice (mlp/conv, trained from scratch). A frozen projection then maps the
    native features to `latent_dim`. When the widths already match, no
    projection is added at all - which is the case that makes the ViT row
    reproduce the S2 reference exactly.
    """
    extra = {"seed": rep_seed} if arch == "random" else {}
    if arch == "random":
        out = latent_dim
    else:
        out = NATIVE_WIDTHS.get(arch, latent_dim)
    base_kwargs: dict = {"input_dim": input_dim, "output_dim": out}
    if arch == "mlp":
        base_kwargs["hidden_dims"] = hidden_dims
    elif arch == "conv":
        base_kwargs["conv_channels"] = conv_channels
    elif arch in RESNET_ARCHES or arch in VIT_ARCHES:
        base_kwargs["backbone_weights"] = backbone_weights
    if input_mean is not None and arch != "random":
        base_kwargs["input_mean"] = input_mean
        base_kwargs["input_std"] = input_std
    base = build_backbone(arch, **base_kwargs, **extra)
    if int(getattr(base, "output_dim", 0)) == int(latent_dim):
        return base
    return ProjectedBackbone(base, out_dim=latent_dim, seed=rep_seed)


_register_shared_encoder_archs()


def wrap_encoder(encoder: nn.Module) -> nn.Module:
    """Adapt an already-built encoder to the `Backbone` interface.

    `CachedFeatureEncoder` and `SharedEncoder` are the two concrete types the
    runner produces; anything else that exposes `output_dim` is wrapped as-is so
    that an external checkpoint (e.g. CLIP/DINO features exported to a tensor
    module) can be plugged in without a new class.
    """
    if isinstance(encoder, CachedFeatureEncoder):
        return CachedBackbone(output_dim=int(encoder.output_dim))
    if isinstance(encoder, SharedEncoder):
        return SharedEncoderBackbone(encoder)
    if hasattr(encoder, "encode") and hasattr(encoder, "output_dim"):
        return encoder
    if hasattr(encoder, "output_dim"):

        class _Wrapped(nn.Module):
            def __init__(self, inner: nn.Module):
                super().__init__()
                self.inner = inner
                self.output_dim = int(inner.output_dim)

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                return self.inner(x)

        return _Wrapped(encoder)
    raise TypeError(f"cannot adapt {type(encoder).__name__} to the Backbone contract")


def optional_backbone(
    arch: str, input_dim: int, output_dim: int, **kwargs
) -> nn.Module | None:
    """Build by name, returning None for an empty name (config convenience)."""
    if not arch:
        return None
    return build_backbone(arch, input_dim=input_dim, output_dim=output_dim, **kwargs)


from .registry import build_backbone  # noqa: E402  (kept at the bottom: no cycles)
