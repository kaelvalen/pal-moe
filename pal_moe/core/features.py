"""Cached frozen-feature tasks: loading, batching, seeding.

Moved verbatim from `experiments/s2_ladder.py` (v3 restructure, phase 1). The
runner re-exports every name, so `s2_ladder.load_tasks` etc. are the same
objects as these.
"""

import numpy as np
import torch

DEFAULT_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"


def load_tasks(cache_path: str):
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = payload["meta"]
    tasks = []
    for row in payload["tasks"]:
        splits = {
            name: (row["splits"][name][0].float(), row["splits"][name][1].long())
            for name in ("train", "val", "test")
        }
        tasks.append(
            {
                "task_id": int(row["task_id"]),
                "classes": [int(c) for c in row["classes"]],
                "splits": splits,
            }
        )
    # Derived, not required: a cache written by a newer or older producer may
    # not carry the dimension in its meta, and the tensors always know it.
    if not meta.get("feature_dim"):
        meta["feature_dim"] = int(tasks[0]["splits"]["train"][0].size(1))
    return meta, tasks



def iter_batches(feats, labels, batch_size, generator=None):
    perm = torch.randperm(feats.size(0), generator=generator)
    for start in range(0, feats.size(0) - batch_size + 1, batch_size):
        idx = perm[start : start + batch_size]
        yield feats[idx], labels[idx]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
