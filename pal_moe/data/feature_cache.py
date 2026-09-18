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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, default_collate


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
        return self.features[index], self.labels[index]


def _collate_features(batch):
    """Batches the stored (possibly half-precision) features and casts once."""
    feats, labels = default_collate(batch)
    return feats.float(), labels


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
    tasks: list[CachedTask]
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
    pin_memory: bool = False,
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

    cached_tasks: list[CachedTask] = []
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
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=_collate_features,
                ),
                val_loader=DataLoader(
                    FeatureTensorDataset(va_f, va_y),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=_collate_features,
                ),
                test_loader=DataLoader(
                    FeatureTensorDataset(te_f, te_y),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=_collate_features,
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


def save_feature_cache(cache: FeatureCache, path: str, meta: dict[str, Any]) -> None:
    """
    Persists a feature cache to `path` (a single .pt file with tensors + meta).

    Used with ``--feature_cache_dir`` so repeated runs of the same encoder (and
    seed-invariant encoders across seeds) skip the feature-extraction pass.
    """
    import os

    payload: dict[str, Any] = {"meta": dict(meta), "tasks": []}
    for task in cache.tasks:
        row: dict[str, Any] = {
            "task_id": task.task_id,
            "classes": list(task.classes) if task.classes is not None else None,
            "splits": {},
        }
        for split, loader in (
            ("train", task.train_loader),
            ("val", task.val_loader),
            ("test", task.test_loader),
        ):
            dataset = loader.dataset
            if not isinstance(dataset, FeatureTensorDataset):  # pragma: no cover
                raise TypeError("feature cache can only be saved from cached tasks")
            row["splits"][split] = (dataset.features, dataset.labels)
        payload["tasks"].append(row)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_feature_cache(
    path: str,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    expected_meta: Optional[dict[str, Any]] = None,
    verbose: bool = True,
) -> Optional[FeatureCache]:
    """
    Loads a persisted feature cache, or returns None when `path` is absent.

    `expected_meta` is compared field-by-field against the stored metadata;
    any mismatch raises, so a stale cache can never silently be reused for a
    different encoder/seed/split.
    """
    import os

    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload.get("meta", {})
    if expected_meta is not None:
        mismatch = {
            key: (meta.get(key), value)
            for key, value in expected_meta.items()
            if meta.get(key) != value
        }
        if mismatch:
            raise ValueError(
                f"feature cache {path} does not match this run "
                f"(field: stored != expected): {mismatch}; "
                "delete the cache or point --feature_cache_dir elsewhere"
            )

    cached_tasks: list[CachedTask] = []
    for row in payload["tasks"]:
        loaders = {}
        for split in ("train", "val", "test"):
            features, labels = row["splits"][split]
            dataset = FeatureTensorDataset(features, labels)
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                drop_last=(split == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=_collate_features,
            )
        cached_tasks.append(
            CachedTask(
                task_id=row["task_id"],
                classes=row["classes"],
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                test_loader=loaders["test"],
            )
        )

    feature_dim = int(
        meta.get("feature_dim", cached_tasks[0].train_loader.dataset.features.size(1))
    )
    if verbose:
        print(
            f"  [feature-cache] loaded {len(cached_tasks)} tasks from {path} "
            f"(feature_dim={feature_dim}, seed={meta.get('seed')})"
        )
    return FeatureCache(
        tasks=cached_tasks,
        encoder=CachedFeatureEncoder(output_dim=feature_dim),
        feature_dim=feature_dim,
    )
