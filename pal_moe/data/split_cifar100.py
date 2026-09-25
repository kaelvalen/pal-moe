"""
Split-CIFAR-100 dataset generator for Continual Learning benchmarks.
Divides CIFAR-100 (100 classes, 20 superclass-free disjoint tasks) into 20
sequential tasks of 5 classes each (standard Split-CIFAR-100 protocol):
  Task k: classes [5k .. 5k+4]
Supports Class-Incremental Learning in a 100-class global space.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass
class SplitCIFAR100Task:
    task_id: int
    classes: tuple[int, ...]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
NUM_TASKS = 20
CLASSES_PER_TASK = 5


def get_split_cifar100_tasks(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_split: float = 0.1,
    seed: int = 42,
    max_train_samples_per_task: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    dataset_cls: Any = None,
) -> list[SplitCIFAR100Task]:
    """
    Creates 20 sequential tasks for the Split-CIFAR-100 benchmark (5 classes each).

    ``num_workers`` / ``pin_memory`` only affect loader throughput; the CIFAR
    transforms are deterministic, so the batch order and RNG stream are
    unchanged when they are raised.

    ``dataset_cls`` exists so a caller can inject a pixel-level distribution
    shift (S9) without re-deriving the split: it must be a `datasets.CIFAR100`
    subclass accepting the same ``transform``/``target_transform`` kwargs. The
    task partition, the index draw and the class order are unchanged, so a
    shifted cache is paired sample-for-sample with the clean one. It is resolved
    here rather than as a default argument, so monkeypatching
    `torchvision.datasets.CIFAR100` still works (the offline loader test does).
    """
    dataset_cls = dataset_cls or datasets.CIFAR100
    torch.manual_seed(seed)
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    train_dataset = dataset_cls(
        data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = dataset_cls(
        data_dir, train=False, download=True, transform=transform_test
    )

    train_targets = torch.tensor(train_dataset.targets)
    test_targets = torch.tensor(test_dataset.targets)

    tasks = []
    for task_id in range(NUM_TASKS):
        classes = tuple(
            range(task_id * CLASSES_PER_TASK, (task_id + 1) * CLASSES_PER_TASK)
        )
        train_mask = torch.isin(train_targets, torch.tensor(classes))
        train_indices = torch.where(train_mask)[0]

        if (
            max_train_samples_per_task is not None
            and len(train_indices) > max_train_samples_per_task
        ):
            perm_all = torch.randperm(len(train_indices))
            train_indices = train_indices[perm_all[:max_train_samples_per_task]]

        num_samples = len(train_indices)
        num_val = int(num_samples * val_split)
        perm = torch.randperm(num_samples)

        val_idx = train_indices[perm[:num_val]]
        tr_idx = train_indices[perm[num_val:]]

        test_mask = torch.isin(test_targets, torch.tensor(classes))
        test_indices = torch.where(test_mask)[0]

        train_loader = DataLoader(
            Subset(train_dataset, tr_idx),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            Subset(train_dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            Subset(test_dataset, test_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        tasks.append(
            SplitCIFAR100Task(
                task_id=task_id,
                classes=classes,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )
        )

    return tasks
