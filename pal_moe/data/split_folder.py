"""
Generic image-folder task splitter.

`Split*` datasets in this package are hardcoded to MNIST/CIFAR. For arbitrary
data ("her türlü veri") this splitter turns any `ImageFolder`-compatible
directory into sequential class-incremental tasks:

    root/
      class_a/*.png
      class_b/*.jpg
      ...

Class order is the folder order (alphabetical), tasks are consecutive groups of
`classes_per_task`, and each class is split deterministically into
train/val/test by `val_split` and `test_split`.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

__all__ = ["FolderTask", "get_split_folder_tasks"]


@dataclass
class FolderTask:
    task_id: int
    classes: tuple
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def get_split_folder_tasks(
    data_dir: str,
    batch_size: int = 128,
    val_split: float = 0.1,
    test_split: float = 0.1,
    classes_per_task: int = 2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    transform: Optional[object] = None,
    max_samples_per_class: Optional[int] = None,
) -> list[FolderTask]:
    """Builds sequential class-incremental tasks from an ImageFolder tree."""
    transform = transform or transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_names = list(dataset.classes)
    if len(class_names) % classes_per_task != 0:
        raise ValueError(
            f"{len(class_names)} classes cannot be split into groups of "
            f"{classes_per_task}; adjust --classes_per_task"
        )

    # Deterministic per-class train/val/test indices.
    torch.manual_seed(seed)
    per_class_indices: dict[int, list[int]] = {c: [] for c in range(len(class_names))}
    for idx, (_, label) in enumerate(dataset.samples):
        per_class_indices[label].append(idx)
    splits = {c: {"train": [], "val": [], "test": []} for c in per_class_indices}
    for c, indices in per_class_indices.items():
        order = torch.randperm(len(indices)).tolist()
        indices = [indices[i] for i in order]
        if max_samples_per_class is not None:
            indices = indices[:max_samples_per_class]
        n = len(indices)
        n_val = int(n * val_split)
        n_test = int(n * test_split)
        splits[c]["val"] = indices[:n_val]
        splits[c]["test"] = indices[n_val : n_val + n_test]
        splits[c]["train"] = indices[n_val + n_test :]

    tasks = []
    for task_id, start in enumerate(range(0, len(class_names), classes_per_task)):
        group = list(range(start, start + classes_per_task))
        loaders = {}
        for split in ("train", "val", "test"):
            idx = [i for c in group for i in splits[c][split]]
            loaders[split] = DataLoader(
                Subset(dataset, idx),
                batch_size=batch_size,
                shuffle=(split == "train"),
                drop_last=(split == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        tasks.append(
            FolderTask(
                task_id=task_id,
                classes=tuple(group),
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                test_loader=loaders["test"],
            )
        )
    return tasks
