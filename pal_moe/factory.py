"""
Shared model factories for the experiment scripts and diagnostics.

Keeping router / model / memory construction in one place stops the benchmark
runner, the exploration sweeps and the checkpoint diagnostics from drifting
apart (the diagnostic could not rebuild AttentionRouter or feature-cached
models before).
"""

import copy
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn

from .data.feature_cache import CachedFeatureEncoder
from .memory.prototype_memory import PrototypeMemory
from .models.encoder import SharedEncoder
from .models.expert import MLPExpert
from .models.moe import DynamicMoE
from .models.router import AttentionRouter, DistanceRouter, DynamicRouter

ROUTER_TYPES = ("dynamic", "distance", "attention")


def build_router(
    router_type: str,
    input_dim: int,
    top_k: int = 1,
    num_experts: int = 1,
    device: Optional[torch.device] = None,
    learn_temperature: bool = False,
):
    """Router factory shared by every runner."""
    if router_type == "distance":
        router = DistanceRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            temperature=0.05,
            learn_temperature=learn_temperature,
        )
    elif router_type == "attention":
        router = AttentionRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            temperature=0.1,
            learn_temperature=learn_temperature,
        )
    elif router_type == "dynamic":
        router = DynamicRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            temperature=1.0,
            learn_temperature=learn_temperature,
        )
    else:
        raise ValueError(
            f"unknown router_type {router_type!r}; use one of {ROUTER_TYPES}"
        )
    return router.to(device) if device is not None else router


def build_prototype_memory(
    feature_dim: int,
    distance_threshold: Optional[float] = 0.5,
    max_prototypes: int = 50,
    max_prototypes_per_class: Optional[int] = None,
    store_raw: bool = False,
    selection: str = "first",
    candidate_pool: int = 4,
    eviction: str = "task",
) -> PrototypeMemory:
    """Prototype-memory factory shared by every runner."""
    return PrototypeMemory(
        feature_dim=feature_dim,
        distance_threshold=distance_threshold,
        ema_alpha=0.9,
        max_prototypes=max_prototypes,
        max_prototypes_per_class=max_prototypes_per_class,
        store_raw=store_raw,
        selection=selection,
        candidate_pool=candidate_pool,
        eviction=eviction,
    )


def build_encoder(
    input_dim: int,
    feature_dim: int,
    arch: str = "mlp",
    hidden_dims: Optional[Sequence[int]] = (256, 128),
    conv_channels: Sequence[int] = (32, 64, 128),
    device: Optional[torch.device] = None,
) -> SharedEncoder:
    encoder = SharedEncoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=feature_dim,
        arch=arch,
        conv_channels=tuple(conv_channels),
    )
    return encoder.to(device) if device is not None else encoder


def build_single_head(
    encoder: nn.Module,
    feature_dim: int,
    expert_hidden: int,
    num_classes: int,
    device: Optional[torch.device] = None,
) -> nn.Sequential:
    """Baseline model: a deep copy of the shared encoder + one matched MLP head."""
    model = nn.Sequential(
        copy.deepcopy(encoder),
        MLPExpert(
            input_dim=feature_dim,
            hidden_dim=expert_hidden,
            num_classes=num_classes,
            expert_id=0,
        ),
    )
    return model.to(device) if device is not None else model


def build_moe(
    encoder: nn.Module,
    feature_dim: int,
    expert_hidden: int,
    num_classes: int,
    num_experts: int = 1,
    router_type: str = "dynamic",
    top_k: int = 1,
    use_ema_encoder: bool = False,
    shared_expert: bool = False,
    device: Optional[torch.device] = None,
) -> DynamicMoE:
    """DynamicMoE factory: router + `num_experts` fresh MLP experts."""
    generalist = (
        MLPExpert(
            input_dim=feature_dim,
            hidden_dim=expert_hidden,
            num_classes=num_classes,
            expert_id=-1,
        )
        if shared_expert
        else None
    )
    model = DynamicMoE(
        encoder=encoder,
        router=build_router(router_type, feature_dim, top_k, num_experts=num_experts),
        experts=[
            MLPExpert(
                input_dim=feature_dim,
                hidden_dim=expert_hidden,
                num_classes=num_classes,
                expert_id=i,
            )
            for i in range(num_experts)
        ],
        use_ema_encoder=use_ema_encoder,
        shared_expert=generalist,
    )
    return model.to(device) if device is not None else model


def build_cached_encoder(feature_dim: int) -> CachedFeatureEncoder:
    """Identity encoder used when running on precomputed frozen features."""
    return CachedFeatureEncoder(feature_dim)
