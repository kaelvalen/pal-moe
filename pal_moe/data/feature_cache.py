"""
Frozen-encoder feature caching.

With a frozen encoder and deterministic input transforms (the benchmark CIFAR/MNIST
pipelines are normalize-only), every stage of the continual pipeline (task
training, candidate training, trigger evaluation, prototype registration, joint
calibration, router distillation and evaluation) only ever needs ``h(x)``, never
``x`` itself. Precomputing ``h`` once per split removes the encoder from the
training loop entirely:

    cache = build_feature_cache(encoder, tasks, device)
    tasks_cached = cache.tasks                  # same interface as Split*Task
    model = DynamicMoE(encoder=cache.encoder, ...)

This is mathematically identical to the raw pipeline (same frozen encoder, same
deterministic transforms; the cached encoder is an identity module), and it
removes all conv forward/backward work from training. CIFAR-10 needs
50k x 256 x 2 bytes ~= 25 MB of feature storage.

RNG note: building the cache iterates the original loaders once, so the global
RNG stream (and therefore the exact batch permutations) differs from a raw run.
Results are distributionally equivalent, not bit-identical; use the raw pipeline
when bit-level comparability matters.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class CachedFeatureEncoder(nn.Module):
    """Identity encoder used in feature-cached runs (the real encoder is frozen)."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = output_dim
        self.arch = "cached"
        self.net = nn.Identity()
        # Device probe: several call sites use next(encoder.parameters()).device,
        # so the identity encoder must expose (non-trainable) parameters.
        self._device_probe = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def freeze(self) -> None:  # pragma: no cover - trivial
        return None

    def unfreeze(self) -> None:  # pragma: no cover - trivial
        return None


class FeatureTensorDataset(Dataset):
    """Dataset of (feature, label) pairs; features are stored in half precision."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert features.size(0) == labels.size(0)
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int):
        return self.features[index].float(), self.labels[index]


@dataclass
class CachedTask:
    """Drop-in replacement for Split*Task backed by cached features."""

    task_id: int
    classes: Any
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


@dataclass
class FeatureCache:
    tasks: List[CachedTask]
    encoder: CachedFeatureEncoder
    feature_dim: int


def _encode_split(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> "tuple[torch.Tensor, torch.Tensor]":
    feats, labels = [], []
    encoder.eval()
    with torch.no_grad():
        for x, y in loader:
            feats.append(encoder(x.to(device)).to(dtype).cpu())
            labels.append(y.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def build_feature_cache(
    encoder: nn.Module,
    tasks: Sequence[Any],
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 128,
    num_workers: int = 0,
    verbose: bool = True,
) -> FeatureCache:
    """
    Precomputes train/val/test features for every task split.

    `encoder` must already be frozen (callers should assert this); the returned
    `FeatureCache.encoder` is an identity module with the right output_dim.
    """
    output_dim: Optional[int] = getattr(encoder, "output_dim", None)
    if output_dim is None:  # pragma: no cover - defensive
        raise ValueError("encoder has no output_dim attribute")

    cached_tasks: List[CachedTask] = []
    for task in tasks:
        tr_f, tr_y = _encode_split(encoder, task.train_loader, device, dtype)
        va_f, va_y = _encode_split(encoder, task.val_loader, device, dtype)
        te_f, te_y = _encode_split(encoder, task.test_loader, device, dtype)
        cached_tasks.append(
            CachedTask(
                task_id=task.task_id,
                classes=task.classes,
                train_loader=DataLoader(
                    FeatureTensorDataset(tr_f, tr_y),
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                ),
                val_loader=DataLoader(
                    FeatureTensorDataset(va_f, va_y),
                    batch_size=batch_size,
                    shuffle=False,
                ),
                test_loader=DataLoader(
                    FeatureTensorDataset(te_f, te_y),
                    batch_size=batch_size,
                    shuffle=False,
                ),
            )
        )
        if verbose:
            size_mb = (
                (tr_f.numel() + va_f.numel() + te_f.numel()) * tr_f.element_size()
            ) / (1024 * 1024)
            print(
                f"  [feature-cache] task {task.task_id}: "
                f"train {tuple(tr_f.shape)} val {tuple(va_f.shape)} test {tuple(te_f.shape)} "
                f"({size_mb:.1f} MB, {str(dtype).replace('torch.', '')})"
            )

    return FeatureCache(
        tasks=cached_tasks,
        encoder=CachedFeatureEncoder(output_dim=int(output_dim)),
        feature_dim=int(output_dim),
    )
