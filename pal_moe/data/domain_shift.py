"""
Domain-shift streams for stress-testing stability.

Class-incremental benchmarks change the label space; production streams more
often keep the classes and change the input distribution (new sensors, seasons,
sites). These helpers wrap an existing task list so every task sees the same
classes under a different deterministic input transform:

- `permutation_transform`: fixed random pixel permutation (a classic
  domain-incremental MNIST variant);
- `rotation_transform`: 90-degree rotations;
- `apply_phase_shift`: applies transform `k` to task `k` (cycling).

The loaders are rebuilt around a transform wrapper, so the underlying datasets
(and their splits) are untouched.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "permutation_transform",
    "rotation_transform",
    "apply_phase_shift",
]


class _TransformedDataset(Dataset):
    def __init__(self, subset: Dataset, transform: Callable):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index):
        x, y = self.subset[index]
        return self.transform(x), y


def permutation_transform(seed: int, input_dim: int) -> Callable:
    """Deterministic pixel permutation (flatten/permute/unflatten)."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(input_dim, generator=gen)

    def _apply(x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1)
        if flat.numel() != input_dim:
            return x
        return flat[perm].reshape(shape)

    return _apply


def rotation_transform(degrees: int) -> Callable:
    """90-degree rotation of a (C, H, W) tensor."""

    def _apply(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            return x
        return torch.rot90(x, k=(degrees // 90) % 4, dims=(1, 2))

    return _apply


@dataclass
class ShiftedTask:
    task_id: int
    classes: tuple
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    domain: int = 0


def _rebuild_loader(loader: DataLoader, dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=isinstance(loader.sampler, torch.utils.data.RandomSampler),
        drop_last=loader.drop_last,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )


def apply_phase_shift(
    tasks: Sequence, transforms: Sequence[Callable]
) -> list[ShiftedTask]:
    """
    Applies `transforms[i % len(transforms)]` to task `i` and returns new tasks.

    Every task gets a `domain` index so evaluation can report per-domain
    accuracy even though the label space is shared.
    """
    shifted: list[ShiftedTask] = []
    for i, task in enumerate(tasks):
        transform = transforms[i % len(transforms)]
        shifted.append(
            ShiftedTask(
                task_id=task.task_id,
                classes=task.classes,
                train_loader=_rebuild_loader(
                    task.train_loader,
                    _TransformedDataset(task.train_loader.dataset, transform),
                ),
                val_loader=_rebuild_loader(
                    task.val_loader,
                    _TransformedDataset(task.val_loader.dataset, transform),
                ),
                test_loader=_rebuild_loader(
                    task.test_loader,
                    _TransformedDataset(task.test_loader.dataset, transform),
                ),
                domain=i % len(transforms),
            )
        )
    return shifted
