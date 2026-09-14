"""
Split-CIFAR-10 dataset generator for Continual Learning benchmarks.
Divides CIFAR-10 into 5 sequential tasks:
Task 0: [0, 1] (airplane, automobile)
Task 1: [2, 3] (bird, cat)
Task 2: [4, 5] (deer, dog)
Task 3: [6, 7] (frog, horse)
Task 4: [8, 9] (ship, truck)
Supports Class-Incremental Learning in a 10-class global space.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class SplitCIFAR10Task:
    task_id: int
    classes: Tuple[int, int]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def get_split_cifar10_tasks(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_split: float = 0.1,
    seed: int = 42,
    max_train_samples_per_task: Optional[int] = None,
) -> List[SplitCIFAR10Task]:
    """
    Creates 5 sequential tasks for Split-CIFAR-10 benchmark.
    """
    torch.manual_seed(seed)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)

    train_targets = torch.tensor(train_dataset.targets)
    test_targets = torch.tensor(test_dataset.targets)

    task_classes = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    tasks = []

    for task_id, (c1, c2) in enumerate(task_classes):
        train_mask = (train_targets == c1) | (train_targets == c2)
        train_indices = torch.where(train_mask)[0]

        if max_train_samples_per_task is not None and len(train_indices) > max_train_samples_per_task:
            perm_all = torch.randperm(len(train_indices))
            train_indices = train_indices[perm_all[:max_train_samples_per_task]]

        num_samples = len(train_indices)
        num_val = int(num_samples * val_split)
        perm = torch.randperm(num_samples)

        val_idx = train_indices[perm[:num_val]]
        tr_idx = train_indices[perm[num_val:]]

        test_mask = (test_targets == c1) | (test_targets == c2)
        test_indices = torch.where(test_mask)[0]

        train_sub = Subset(train_dataset, tr_idx)
        val_sub = Subset(train_dataset, val_idx)
        test_sub = Subset(test_dataset, test_indices)

        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_sub, batch_size=batch_size, shuffle=False)

        tasks.append(
            SplitCIFAR10Task(
                task_id=task_id,
                classes=(c1, c2),
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )
        )

    return tasks
