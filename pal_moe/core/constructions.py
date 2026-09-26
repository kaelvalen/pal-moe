"""S6b task constructions (`coherent` / `dispersed`) over a cached CIFAR-100 source.

Moved verbatim from `experiments/s6b_difficulty.py` (v3 restructure, phase 1).
"""

import os
import statistics

import torch


def superclass_of(data_dir: str = "./data") -> dict[int, int]:
    """CIFAR-100 fine class -> coarse (super) class.

    Read from the raw pickle rather than the torchvision wrapper: the wrapper
    does not expose `coarse_labels` in this version, and the mapping is what the
    whole construction rests on.
    """
    import pickle

    path = os.path.join(data_dir, "cifar-100-python", "train")
    with open(path, "rb") as fh:
        payload = pickle.load(fh, encoding="bytes")
    mapping: dict[int, int] = {}
    conflicts = 0
    for fine, coarse in zip(payload[b"fine_labels"], payload[b"coarse_labels"]):
        fine, coarse = int(fine), int(coarse)
        if fine in mapping and mapping[fine] != coarse:
            conflicts += 1
        mapping[fine] = coarse
    if len(mapping) != 100 or conflicts:
        raise RuntimeError(
            f"bad fine->coarse mapping: {len(mapping)} classes, {conflicts} conflicts"
        )
    return mapping


def args_data_dir() -> str:
    """The data root the source cache was built from (config default)."""
    return os.environ.get("PAL_MOE_DATA_DIR", "./data")


def build_construction(tasks: list[dict], construct: str) -> list[dict]:
    """Regroup the cached samples into tasks; features are untouched."""
    per_task = len(tasks[0]["classes"])
    feats = torch.cat([t["splits"]["train"][0] for t in tasks], dim=0)
    labels = torch.cat([t["splits"]["train"][1] for t in tasks], dim=0)
    test_feats = torch.cat([t["splits"]["test"][0] for t in tasks], dim=0)
    test_labels = torch.cat([t["splits"]["test"][1] for t in tasks], dim=0)

    mapping = superclass_of(args_data_dir())
    classes = sorted(mapping)
    if construct == "coherent":
        # One task per superclass: its five fine classes together.
        groups = [
            [c for c in classes if mapping[c] == coarse]
            for coarse in sorted(set(mapping.values()))
        ]
    elif construct == "dispersed":
        # Round-robin in superclass order: each task draws one class from each
        # of five well-separated superclasses.
        ordered = sorted(classes, key=lambda c: (mapping[c], c))
        n_tasks = len(classes) // per_task
        groups = [[] for _ in range(n_tasks)]
        for index, c in enumerate(ordered):
            groups[index % n_tasks].append(c)
    else:
        raise KeyError(f"unknown construction {construct!r}")

    rebuilt = []
    for index, group in enumerate(groups):
        assert len(group) == per_task, (index, group)
        train_mask = torch.isin(labels, torch.tensor(group))
        test_mask = torch.isin(test_labels, torch.tensor(group))
        rebuilt.append(
            {
                "task_id": index,
                "classes": list(group),
                "splits": {
                    "train": (feats[train_mask], labels[train_mask]),
                    "val": (test_feats[test_mask], test_labels[test_mask]),
                    "test": (test_feats[test_mask], test_labels[test_mask]),
                },
            }
        )
    return rebuilt


@torch.no_grad()
def separability(tasks: list[dict]) -> dict:
    """Intra- and cross-task class-mean similarity, and their difference."""
    means = []
    for task in tasks:
        feats, labels = task["splits"]["train"]
        per_class = []
        for c in task["classes"]:
            mask = labels == c
            if bool(mask.any()):
                per_class.append(feats[mask].mean(dim=0))
        if per_class:
            means.append(torch.nn.functional.normalize(torch.stack(per_class), dim=-1))

    intra = []
    for block in means:
        if block.size(0) < 2:
            continue
        sim = block @ block.t()
        off = sim[~torch.eye(block.size(0), dtype=torch.bool)]
        intra.append(float(off.max()))  # nearest neighbour inside the task

    cross = []
    for i in range(len(means)):
        for j in range(len(means)):
            if i == j:
                continue
            cross.append(float((means[i] @ means[j].t()).max()))

    intra_mean = statistics.mean(intra) if intra else float("nan")
    cross_mean = statistics.mean(cross) if cross else float("nan")
    return {
        "intra_task_similarity": intra_mean,
        "cross_task_similarity": cross_mean,
        "separability": intra_mean - cross_mean,
    }
