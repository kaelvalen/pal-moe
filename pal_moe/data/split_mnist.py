"""
Split-MNIST dataset generator for Continual Learning benchmarks.
Divides MNIST into 5 sequential tasks: [0,1], [2,3], [4,5], [6,7], [8,9].
Supports Class-Incremental Learning (10-class global space).
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class SplitMNISTTask:
    task_id: int
    classes: Tuple[int, int]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def get_split_mnist_tasks(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> List[SplitMNISTTask]:
    """
    Creates 5 sequential tasks for Split-MNIST benchmark:
    Task 0: digits 0, 1
    Task 1: digits 2, 3
    Task 2: digits 4, 5
    Task 3: digits 6, 7
    Task 4: digits 8, 9
    """
    torch.manual_seed(seed)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    task_classes = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    tasks = []

    for task_id, (c1, c2) in enumerate(task_classes):
        # Filter train indices
        train_targets = train_dataset.targets
        train_mask = (train_targets == c1) | (train_targets == c2)
        train_indices = torch.where(train_mask)[0]

        # Train / Val split
        num_samples = len(train_indices)
        num_val = int(num_samples * val_split)
        perm = torch.randperm(num_samples)

        val_idx = train_indices[perm[:num_val]]
        tr_idx = train_indices[perm[num_val:]]

        # Filter test indices
        test_targets = test_dataset.targets
        test_mask = (test_targets == c1) | (test_targets == c2)
        test_indices = torch.where(test_mask)[0]

        train_sub = Subset(train_dataset, tr_idx)
        val_sub = Subset(train_dataset, val_idx)
        test_sub = Subset(test_dataset, test_indices)

        train_loader = DataLoader(
            train_sub,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_sub,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_sub,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        tasks.append(
            SplitMNISTTask(
                task_id=task_id,
                classes=(c1, c2),
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )
        )

    return tasks
