"""
S4 - dataset generalization under one fixed backbone.

Question: do the S2/S3 behaviours survive a change of dataset?

    G1 readout dominance    is `ridge > L3` true on most datasets?
    G2 isolation benefit    is `L3 - L2b` positive, and does it survive?
    G3 routing tax          how does `L4 - L3` move with the dataset?
    G4 representation dependence   does transfer correlate with `L3 - ridge`?
    G5 low-data penalty     is the few-shot delta systematically worse for the
                            per-task bank, as S7 found on three backbones?
    G6 scaling              memory / params / latency vs task and class count

Canonical backbone: **frozen ViT-B/16 ImageNet**, latent_dim 768, so the
S2 -> S3 -> S4 chain stays comparable. Only the dataset changes. Each dataset
keeps the repo's canonical split (MNIST 5x2, CIFAR-10 5x2, CIFAR-100 20x5,
Tiny-ImageNet 20x10), which is what makes G6 measurable.

Every cell reports the S0 contract blocks it can satisfy, and the transfer suite
(full + 1/5/10-shot, against the raw frozen reference) because S7 showed that
the full split alone hides the adaptation's low-data cost.

Usage:
    python experiments/s4_datasets.py --device cuda
    python experiments/s4_datasets.py --only cifar10 --seeds 42
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision import transforms  # noqa: E402

import s2_ladder  # noqa: E402
import s7_transfer  # noqa: E402
from pal_moe.arch import build_projected_backbone  # noqa: E402
from pal_moe.data.feature_cache import (
    build_feature_cache,
    save_feature_cache,
)  # noqa: E402
from pal_moe.data.split_cifar import get_split_cifar10_tasks  # noqa: E402
from pal_moe.data.split_cifar100 import get_split_cifar100_tasks  # noqa: E402
from pal_moe.data.split_folder import get_split_folder_tasks  # noqa: E402
from pal_moe.data.split_mnist import get_split_mnist_tasks  # noqa: E402
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402

DATASET_NORM = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

# The S4 main ladder: no diagnostic readouts, no joint-probe control.
LEVELS = [
    "L0_ncm",
    "L1_ridge",
    "L2a_shared_joint",
    "L2b_shared_seq",
    "L3_per_task",
    "L4_oracle",
]

# Canonical backbone for the whole stage.
BACKBONE = dict(arch="vit_b_16", weights="imagenet", latent_dim=768)

DATASETS = {
    "mnist": {"kind": "mnist", "classes_per_task": 2, "num_tasks": 5},
    "cifar10": {"kind": "cifar10", "classes_per_task": 2, "num_tasks": 5},
    "cifar100": {"kind": "cifar100", "classes_per_task": 5, "num_tasks": 20},
    "tinyimagenet": {
        "kind": "folder",
        "data_dir": "data/tiny-imagenet-200/train_flat",
        "classes_per_task": 10,
        "num_tasks": 20,
        "image_size": 224,
    },
}
# S4a: one fixed cross-dataset probe. CIFAR-10 is the only source for which
# this is in-domain, and that is recorded rather than hidden.
DOWNSTREAM = "cifar10"


def build_tasks(spec: dict, args, seed: int):
    common = dict(
        batch_size=args.batch_size,
        seed=seed,
        num_workers=args.num_workers,
        val_split=0.1,
    )
    kind = spec["kind"]
    if kind == "mnist":
        return get_split_mnist_tasks(data_dir=args.data_dir, **common)
    if kind == "cifar10":
        return get_split_cifar10_tasks(data_dir=args.data_dir, **common)
    if kind == "cifar100":
        return get_split_cifar100_tasks(data_dir=args.data_dir, **common)
    if kind == "folder":
        size = spec.get("image_size", 224)
        transform = transforms.Compose(
            [transforms.Resize((size, size)), transforms.ToTensor()]
        )
        return get_split_folder_tasks(
            data_dir=spec["data_dir"],
            test_split=0.1,
            classes_per_task=spec["classes_per_task"],
            pin_memory=False,
            transform=transform,
            **common,
        )
    raise KeyError(f"unknown dataset kind {kind!r}")


def ensure_cache(name: str, spec: dict, args, device) -> str:
    """Build (or reuse) the frozen-backbone feature cache for one dataset."""
    path = os.path.join(args.out, f"cache_{name}", "feature_cache.pt")
    if os.path.exists(path):
        print(f"[S4] {name}: cache exists", flush=True)
        return path
    mean, std = DATASET_NORM.get(spec["kind"], (None, None))
    backbone = build_projected_backbone(
        BACKBONE["arch"],
        input_dim=3 * 32 * 32,
        latent_dim=BACKBONE["latent_dim"],
        backbone_weights=BACKBONE["weights"],
        input_mean=mean,
        input_std=std,
    ).to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    tasks = build_tasks(spec, args, seed=42)
    cache = build_feature_cache(
        backbone, tasks, device, num_workers=args.num_workers, verbose=False
    )
    save_feature_cache(
        cache,
        path,
        meta={
            "dataset": name,
            "encoder_arch": BACKBONE["arch"],
            "encoder_weights": BACKBONE["weights"],
            "feature_dim": int(cache.feature_dim),
            "latent_dim": BACKBONE["latent_dim"],
            "rep_seed": 0,
            "num_tasks": len(tasks),
            "projected": True,
            "seed": None,
        },
    )
    print(f"[S4] {name}: cache built ({len(tasks)} tasks)", flush=True)
    return path


def run_cell(
    name: str, spec: dict, level: str, source_cache: str, downstream, args, device
):
    meta, tasks = s2_ladder.load_tasks(source_cache)
    dim = int(meta["feature_dim"])
    num_classes = sum(len(t["classes"]) for t in tasks)
    model_spec = s2_ladder.LEVELS_BY_NAME[level]

    raw = s7_transfer.transfer_suite(
        downstream["train"][0],
        downstream["train"][1],
        downstream["test"][0],
        downstream["test"][1],
        args.seed,
    )

    s2_ladder.set_seed(args.seed)
    model = s2_ladder.LadderModel(model_spec, dim, num_classes, args, device)
    n = len(tasks)
    R = np.zeros((n, n), dtype=np.float32)
    fwt = []
    for t, task in enumerate(tasks):
        if model.seen:
            _adapted, delta = s2_ladder.forward_transfer(model, task, num_classes, dim)
            fwt.append(delta)
        model.seen = sorted(set(model.seen) | set(task["classes"]))
        model.fit_task(task, t)
        model.register_task(task, t)
        for i in range(t + 1):
            R[t, i] = model.evaluate_task(tasks[i])

    T = n - 1
    forget = [
        max(0.0, float(np.max(R[i : T + 1, i])) - float(R[T, i])) for i in range(T)
    ]
    with torch.no_grad():
        z_tr = model.represent(downstream["train"][0].to(device)).cpu()
        z_te = model.represent(downstream["test"][0].to(device)).cpu()
    transfer = s7_transfer.transfer_suite(
        z_tr, downstream["train"][1], z_te, downstream["test"][1], args.seed
    )
    routing = model.routing_stats(tasks)
    cost = model.cost()
    return {
        "dataset": name,
        "level": level,
        "seed": args.seed,
        "in_domain_transfer": name == DOWNSTREAM,
        "num_tasks": n,
        "classes_per_task": len(tasks[0]["classes"]),
        "accuracy": float(np.mean(R[T, :])),
        "forgetting": float(np.mean(forget)) if forget else 0.0,
        "bwt": float(np.mean([R[T, i] - R[i, i] for i in range(T)])) if T else 0.0,
        "fwt": float(np.mean(fwt)) if fwt else None,
        "acc_matrix": R.tolist(),
        "raw_transfer": raw,
        "transfer": transfer,
        "delta_transfer": {k: transfer[k] - raw[k] for k in transfer},
        "oracle_accuracy": (
            float(np.mean(R[T, :])) if model_spec.router == "oracle" else None
        ),
        "params": cost["params"],
        "flops_forward": cost["flops_forward"],
        "optimizer_steps": cost["optimizer_steps"],
        "expert_count": 0 if not model_spec.expert else n,
        **routing,
    }


def aggregate(cells: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = {}
    for cell in cells:
        grouped.setdefault((cell["dataset"], cell["level"]), []).append(cell)

    def st(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        if len(values) == 1:
            return {"mean": float(values[0]), "std": 0.0, "n": 1}
        return {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.stdev(values)),
            "n": len(values),
        }

    matrix = {}
    for (dataset, level), rows in grouped.items():
        matrix.setdefault(dataset, {})[level] = {
            "accuracy": st([r["accuracy"] for r in rows]),
            "forgetting": st([r["forgetting"] for r in rows]),
            "fwt": st([r["fwt"] for r in rows]),
            "transfer_full": st([r["transfer"]["full"] for r in rows]),
            "transfer_few1": st([r["transfer"]["few1"] for r in rows]),
            "transfer_few5": st([r["transfer"]["few5"] for r in rows]),
            "transfer_few10": st([r["transfer"]["few10"] for r in rows]),
            "delta_few1": st([r["delta_transfer"]["few1"] for r in rows]),
            "delta_few10": st([r["delta_transfer"]["few10"] for r in rows]),
            "routing_tax": None,
            "params": rows[0]["params"],
            "num_tasks": rows[0]["num_tasks"],
            "classes_per_task": rows[0]["classes_per_task"],
            "seeds": [r["seed"] for r in rows],
        }

    deltas = {}
    for dataset, levels in matrix.items():
        d = {}
        if "L1_ridge" in levels and "L3_per_task" in levels:
            d["L3_minus_ridge"] = (
                levels["L3_per_task"]["accuracy"]["mean"]
                - levels["L1_ridge"]["accuracy"]["mean"]
            )
        if "L2b_shared_seq" in levels and "L3_per_task" in levels:
            d["isolation_benefit_L3_minus_L2b"] = (
                levels["L3_per_task"]["accuracy"]["mean"]
                - levels["L2b_shared_seq"]["accuracy"]["mean"]
            )
        if "L3_per_task" in levels and "L4_oracle" in levels:
            d["routing_tax_L4_minus_L3"] = (
                levels["L4_oracle"]["accuracy"]["mean"]
                - levels["L3_per_task"]["accuracy"]["mean"]
            )
            levels["L3_per_task"]["routing_tax"] = d["routing_tax_L4_minus_L3"]
        if "L2a_shared_joint" in levels and "L1_ridge" in levels:
            d["L2a_minus_ridge"] = (
                levels["L2a_shared_joint"]["accuracy"]["mean"]
                - levels["L1_ridge"]["accuracy"]["mean"]
            )
        deltas[dataset] = d
    return {"matrix": matrix, "deltas": deltas}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="")
    parser.add_argument("--levels", nargs="*", default=LEVELS)
    parser.add_argument("--seeds", default="42,1,2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument("--max_experts", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/s4")
    parser.add_argument("--skip_extraction", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore recorded cells")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    wanted = {s for s in args.only.split(",") if s}
    names = [n for n in DATASETS if not wanted or n in wanted]
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]

    if not args.skip_extraction:
        for name in names:
            ensure_cache(name, DATASETS[name], args, device)

    downstream_cache = os.path.join(args.out, f"cache_{DOWNSTREAM}", "feature_cache.pt")
    if not os.path.exists(downstream_cache):
        downstream_cache = (
            f"results/feature_cache/{DOWNSTREAM}_vit_b16/feature_cache.pt"
        )
    downstream = s7_transfer.load_downstream(downstream_cache)
    print(
        f"[S4] downstream={DOWNSTREAM} ({downstream['train'][0].shape[0]} train rows), "
        f"backbone={BACKBONE['arch']}/{BACKBONE['weights']}",
        flush=True,
    )

    # Resume: a server restart during a multi-hour matrix must not throw away
    # completed cells. The study file is rewritten after every cell.
    study_path = os.path.join(args.out, "s4_dataset_study.json")
    cells: list[dict] = []
    done: set[tuple] = set()
    # The resume key must include the recipe: skipping "already recorded" cells
    # that were produced with different epochs/lr/rank would silently mix
    # incompatible rows into one matrix.
    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "rank": args.rank,
        "lambda_func": args.lambda_func,
        "levels": list(args.levels),
    }
    if os.path.exists(study_path) and not args.force:
        try:
            previous = json.load(open(study_path))
            if previous.get("recipe") != recipe:
                print(
                    "[S4] recorded study was produced with a different recipe "
                    f"({previous.get('recipe')} != {recipe}); starting fresh",
                    flush=True,
                )
                cells = []
            else:
                cells = [
                    c
                    for c in previous.get("cells", [])
                    if not wanted or c["dataset"] in wanted
                ]
                print(f"[S4] resuming: {len(cells)} cells already recorded", flush=True)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[S4] could not resume ({exc}), starting fresh", flush=True)
            cells = []
    done: set[tuple] = {(c["dataset"], c["level"], c["seed"]) for c in cells}

    def _save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "s4_dataset_generalization",
            "backbone": BACKBONE,
            "downstream": DOWNSTREAM,
            "protocol": "class_il",
            "seeds": seeds,
            "recipe": recipe,
            "cells": cells,
            "aggregate": aggregate(cells) if cells else {},
            "contracts": {
                f"{c['dataset']}__{c['level']}__seed{c['seed']}": _contract(c)
                for c in cells
            },
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)

    for name in names:
        source = os.path.join(args.out, f"cache_{name}", "feature_cache.pt")
        if not os.path.exists(source):
            print(f"[S4] {name}: no cache, skipping", flush=True)
            continue
        for level in args.levels:
            for seed in seeds:
                if (name, level, seed) in done:
                    continue
                args.seed = seed
                cell = run_cell(
                    name, DATASETS[name], level, source, downstream, args, device
                )
                cells.append(cell)
                _save()
                print(
                    f"[S4] {name:13s} {level:16s} seed={seed:<3d} "
                    f"acc={cell['accuracy'] * 100:6.2f}%  "
                    f"F={cell['forgetting'] * 100:5.2f}%  "
                    f"tr={cell['transfer']['full'] * 100:5.2f}%  "
                    f"few1={cell['transfer']['few1'] * 100:5.2f}%",
                    flush=True,
                )

    _save()
    print(f"[S4] wrote {study_path}", flush=True)
    _print_matrix(aggregate(cells))


def _contract(cell: dict) -> dict:
    record = build_run_record(
        factors={
            "dataset": cell["dataset"],
            "protocol": "class_il",
            "task_id_at_inference": cell["level"] == "L4_oracle",
            "num_tasks": cell["num_tasks"],
            "classes_per_task": cell["classes_per_task"],
            "seed": cell["seed"],
            "model_family": cell["level"],
            "backbone": f"{BACKBONE['arch']}+proj{BACKBONE['latent_dim']}",
            "backbone_pretraining": "imagenet_frozen",
            "readout": "ncm" if cell["level"] == "L0_ncm" else "ridge",
            "readout_estimator": "offline_mean" if cell["level"] == "L0_ncm" else None,
            "expert": (
                "none"
                if cell["level"] in ("L0_ncm", "L1_ridge")
                else "residual_adapter"
            ),
        },
        metrics={
            "learning": {
                "accuracy": cell["accuracy"],
                "forgetting": cell["forgetting"],
                "bwt": cell["bwt"],
                "fwt": cell["fwt"],
                "acc_matrix": cell["acc_matrix"],
            },
            "cost": {
                "stored_bytes": _stored_bytes(cell),
                "total_params": cell["params"],
                "flops_forward": cell["flops_forward"],
                "optimizer_steps": cell["optimizer_steps"],
                "latency_ms": None,
            },
            "generalization": {
                "transfer_accuracy": cell["transfer"]["full"],
                "unseen_task_accuracy": None,
            },
            "stability": None,
            "modular": {
                "expert_count": cell["expert_count"],
                "routing_entropy": None,
                "utilization": None,
                "task_recall_at_k": (
                    {"3": cell["task_recall_at_3"]}
                    if "task_recall_at_3" in cell
                    else None
                ),
                "oracle_accuracy": cell["oracle_accuracy"],
            },
        },
        provenance={
            "command": "experiments/s4_datasets.py",
            "raw_transfer": cell["raw_transfer"],
            "delta_transfer": cell["delta_transfer"],
            "in_domain_transfer": cell["in_domain_transfer"],
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


def _stored_bytes(cell: dict) -> int:
    """Prototype store: the class means the NCM readout or the router keeps."""
    dim = BACKBONE["latent_dim"]
    n_classes = cell["num_tasks"] * cell["classes_per_task"]
    if cell["level"] == "L0_ncm" or "task_recall_at_3" in cell:
        return int(n_classes * dim * 4)
    return 0


def _print_matrix(agg: dict) -> None:
    print("\n" + "=" * 104)
    print("S4 DATASET x COMPLEXITY (frozen ViT-B/16, Class-IL, mean over seeds)")
    print("=" * 104)
    print(
        f"{'dataset':14s} {'tasks':>5s} {'NCM':>8s} {'Ridge':>8s} {'Shared-J':>9s} "
        f"{'Shared-S':>9s} {'PerTask':>8s} {'Oracle':>8s} {'tax':>7s} {'few1 d':>8s}"
    )
    for dataset, levels in agg["matrix"].items():

        def g(level, key="accuracy", levels=levels):
            v = levels.get(level, {}).get(key)
            return v["mean"] if v else None

        def p(x):
            return f"{x * 100:7.2f}%" if x is not None else "      -"

        first = next(iter(levels.values()))
        print(
            f"{dataset:14s} {first['num_tasks']:5d} {p(g('L0_ncm'))} {p(g('L1_ridge'))} "
            f"{p(g('L2a_shared_joint'))} {p(g('L2b_shared_seq'))} {p(g('L3_per_task'))} "
            f"{p(g('L4_oracle'))} "
            f"{p(levels.get('L3_per_task', {}).get('routing_tax'))} "
            f"{p(g('L3_per_task', 'delta_few1'))}"
        )
    print("\ndeltas")
    for dataset, d in agg["deltas"].items():
        print(
            f"  {dataset:14s} "
            + "  ".join(f"{k}={v * 100:+.2f}%" for k, v in d.items())
        )


if __name__ == "__main__":
    main()
