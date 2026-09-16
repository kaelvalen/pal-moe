"""
Post-hoc diagnostics for a saved PAL-MoE checkpoint.

Separates the two possible causes of catastrophic forgetting:

* **router failure** — the experts still solve their own tasks, but the router
  never selects them for old inputs (the "recency funnel"), vs.
* **destructive calibration** — the experts themselves no longer solve their
  tasks because end-of-task joint fine-tuning overwrote them.

For every (task, expert) pair the script reports:

1. the expert's accuracy on that task's test set (features from the checkpoint's
   encoder), i.e. "what would this expert say?",
2. the router's top-1 assignment share for that task, i.e. "who does the router
   actually pick?",
3. the end-to-end accuracy and the per-task oracle (best expert) accuracy.

Usage:
    python experiments/diagnose_checkpoint.py \
        --checkpoint results/cifar10_big/checkpoints_palmoe/task_4.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tabulate import tabulate

from pal_moe.data.split_cifar import get_split_cifar10_tasks
from pal_moe.data.split_cifar100 import get_split_cifar100_tasks
from pal_moe.data.split_mnist import get_split_mnist_tasks
from pal_moe.memory.prototype_memory import PrototypeMemory
from pal_moe.models.encoder import SharedEncoder
from pal_moe.models.expert import MLPExpert
from pal_moe.models.moe import DynamicMoE
from pal_moe.models.router import DistanceRouter, DynamicRouter
from pal_moe.persistence import load_checkpoint

INPUT_DIMS = {"mnist": 784, "cifar10": 3072, "cifar100": 3072}


def infer_config(state, dataset):
    """Recover model geometry from a checkpoint's state dict."""
    expert_ids = sorted(
        {int(k.split(".")[1]) for k in state if k.startswith("experts.")}
    )
    num_experts = max(expert_ids) + 1
    expert_hidden, feature_dim = state["experts.0.fc1.weight"].shape
    num_classes = state["experts.0.fc2.weight"].shape[0]

    conv_ws = sorted(
        (int(k.split(".")[2]), v.shape[0])
        for k, v in state.items()
        if k.startswith("encoder.net.") and k.endswith(".weight") and v.dim() == 4
    )
    if conv_ws:
        arch, hidden_dims = "conv", None
        conv_channels = tuple(ch for _, ch in conv_ws)
    else:
        lin_ws = sorted(
            (int(k.split(".")[2]), v.shape[0])
            for k, v in state.items()
            if k.startswith("encoder.net.") and k.endswith(".weight") and v.dim() == 2
        )
        arch, hidden_dims = "mlp", tuple(ch for _, ch in lin_ws[:-1])
        conv_channels = (32, 64, 128)  # unused for the MLP architecture

    return {
        "num_experts": num_experts,
        "feature_dim": int(feature_dim),
        "expert_hidden": int(expert_hidden),
        "num_classes": int(num_classes),
        "arch": arch,
        "hidden_dims": hidden_dims,
        "conv_channels": conv_channels,
        "input_dim": INPUT_DIMS[dataset],
        "is_distance_router": any(k.startswith("router.centroids") for k in state),
    }


def build_model(cfg, device):
    encoder = SharedEncoder(
        input_dim=cfg["input_dim"],
        hidden_dims=cfg["hidden_dims"],
        output_dim=cfg["feature_dim"],
        arch=cfg["arch"],
        conv_channels=cfg["conv_channels"],
    )
    if cfg["is_distance_router"]:
        router = DistanceRouter(
            input_dim=cfg["feature_dim"], num_experts=cfg["num_experts"], top_k=1
        )
    else:
        router = DynamicRouter(
            input_dim=cfg["feature_dim"], num_experts=cfg["num_experts"], top_k=1
        )
    experts = [
        MLPExpert(
            input_dim=cfg["feature_dim"],
            hidden_dim=cfg["expert_hidden"],
            num_classes=cfg["num_classes"],
            expert_id=i,
        )
        for i in range(cfg["num_experts"])
    ]
    model = DynamicMoE(encoder=encoder, router=router, experts=experts)
    model.to(device)
    model.eval()
    return model


def load_tasks(dataset, batch_size, num_workers):
    kwargs = dict(batch_size=batch_size, val_split=0.1, seed=42, num_workers=num_workers)
    if dataset == "cifar10":
        return get_split_cifar10_tasks(data_dir="./data", **kwargs)
    if dataset == "cifar100":
        return get_split_cifar100_tasks(data_dir="./data", **kwargs)
    return get_split_mnist_tasks(data_dir="./data", **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to task_*.pt")
    parser.add_argument(
        "--dataset", default="cifar10", choices=["mnist", "cifar10", "cifar100"]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Diagnose] device={device} checkpoint={args.checkpoint}")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    cfg = infer_config(state, args.dataset)
    print(
        f"[Diagnose] experts={cfg['num_experts']} feature_dim={cfg['feature_dim']} "
        f"expert_hidden={cfg['expert_hidden']} arch={cfg['arch']} "
        f"conv_channels={cfg['conv_channels']}"
    )

    model = build_model(cfg, device)
    memory = PrototypeMemory(feature_dim=cfg["feature_dim"], store_raw=False)
    meta = load_checkpoint(args.checkpoint, model, memory, device=device)
    print(f"[Diagnose] checkpoint meta: {meta}")

    tasks = load_tasks(args.dataset, args.batch_size, args.num_workers)
    num_experts = cfg["num_experts"]
    n = len(tasks)

    acc_mat = np.zeros((n, num_experts))
    share_mat = np.zeros((n, num_experts))
    e2e = np.zeros(n)
    anchored_e2e = np.zeros(n)
    coverage = np.zeros(n)
    owner_readout = np.zeros(n)

    model.set_prototype_routing(memory, alpha=1.0)

    with torch.no_grad():
        for j, task in enumerate(tasks):
            feats, labels = [], []
            correct = anchored_correct = total = 0
            for x, y in task.test_loader:
                x, y = x.to(device), y.to(device)
                h = model.encoder(x)
                feats.append(h)
                labels.append(y)
                model.proto_routing_alpha = 0.0
                correct += (model(x).argmax(dim=1) == y).sum().item()
                model.proto_routing_alpha = 1.0
                anchored_correct += (model(x).argmax(dim=1) == y).sum().item()
                total += y.numel()
            h = torch.cat(feats)
            y = torch.cat(labels)
            e2e[j] = correct / max(total, 1)
            anchored_e2e[j] = anchored_correct / max(total, 1)

            if not memory.is_empty():
                v_mat = memory.get_prototype_matrix(device)
                dists = torch.cdist(h, v_mat)
                min_dist = dists.min(dim=1).values
                threshold = (
                    model.proto_routing_threshold
                    if model.proto_routing_threshold is not None
                    else memory.distance_threshold
                )
                coverage[j] = (min_dist <= threshold).float().mean().item()
                q = torch.quantile(
                    min_dist, torch.tensor([0.1, 0.5, 0.9], device=min_dist.device)
                )
                print(
                    f"  [dist] task {j}: min-dist p10/p50/p90 = "
                    f"{float(q[0]):.2f} / {float(q[1]):.2f} / {float(q[2]):.2f} "
                    f"(threshold {threshold})"
                )

                # Owner-expert readout: treat the nearest prototype's owner expert
                # (the expert trained for that task) as the classifier.
                nearest = dists.argmin(dim=1)
                owner_ids = torch.tensor(
                    [
                        p.owner_expert if p.owner_expert is not None else -1
                        for p in memory.prototypes
                    ],
                    device=device,
                    dtype=torch.long,
                )
                owners = owner_ids[nearest]
                valid = owners >= 0
                if valid.any():
                    correct_owner = 0
                    for eid in owners[valid].unique().tolist():
                        if eid >= num_experts:
                            continue
                        m = valid & (owners == eid)
                        logits = model.experts[eid](h[m])
                        correct_owner += (
                            logits.argmax(dim=1) == y[m]
                        ).sum().item()
                    owner_readout[j] = correct_owner / max(total, 1)

            for i, expert in enumerate(model.experts):
                pred = expert(h).argmax(dim=1)
                acc_mat[j, i] = (pred == y).float().mean().item()

            _, topk, _ = model.router(h)
            counts = torch.bincount(topk.flatten(), minlength=num_experts).float()
            share_mat[j] = (counts / counts.sum()).cpu().numpy()

    print("\nExpert accuracy on each task (row = task, col = expert):")
    print(
        tabulate(
            [[f"Task {j}"] + [f"{acc_mat[j, i]:6.1%}" for i in range(num_experts)] for j in range(n)],
            headers=[""] + [f"E{i}" for i in range(num_experts)],
            tablefmt="github",
        )
    )
    print("\nRouter top-1 share for each task (row = task, col = expert):")
    print(
        tabulate(
            [[f"Task {j}"] + [f"{share_mat[j, i]:6.1%}" for i in range(num_experts)] for j in range(n)],
            headers=[""] + [f"E{i}" for i in range(num_experts)],
            tablefmt="github",
        )
    )

    oracle = acc_mat.max(axis=1)
    best = acc_mat.argmax(axis=1)
    print("\nPer-task summary:")
    print(
        tabulate(
            [
                [
                    f"Task {j}",
                    f"{e2e[j]:6.1%}",
                    f"{anchored_e2e[j]:6.1%}",
                    f"{oracle[j]:6.1%}",
                    f"E{int(best[j])} ({oracle[j]:.1%})",
                    f"E{int(share_mat[j].argmax())}",
                    f"{coverage[j]:6.1%}",
                    f"{owner_readout[j]:6.1%}",
                ]
                for j in range(n)
            ],
            headers=[
                "",
                "router e2e",
                "anchored e2e",
                "oracle (best expert)",
                "best expert",
                "routed to",
                "proto coverage",
                "owner readout",
            ],
            tablefmt="github",
        )
    )
    print(
        f"\n[Diagnose] mean router e2e={e2e.mean():.1%} | "
        f"mean anchored e2e={anchored_e2e.mean():.1%} | "
        f"mean oracle={oracle.mean():.1%} | "
        f"mean owner readout={owner_readout.mean():.1%}"
    )


if __name__ == "__main__":
    main()
