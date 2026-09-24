"""
S5 - the protocol axis: Class-IL, Task-IL, and the two mechanisms behind the
difference.

The v1 codebase conflated two different things. **Oracle routing is not
Task-IL**: oracle routing hands the model the task id *for routing* while it
still chooses among every class it has seen; Task-IL additionally restricts the
class search space to the task's own classes. Both change the accuracy, for
different reasons, so S5 measures the 2x2 factorial:

    class_masking=False, routing=learned   Class-IL (the reference)
    class_masking=False, routing=oracle    task id for routing only
    class_masking=True,  routing=learned   class space restricted only
    class_masking=True,  routing=oracle    Task-IL (the standard protocol)

The interesting quantity is therefore not `L4 - L3` alone but the pair

    Acc(masking) - Acc(class_il)        the class-space contribution
    Acc(oracle)  - Acc(learned)         the routing contribution

recorded per level, because only the per-task levels (L3, L4) make a routing
decision at all - L0/L1 have no expert and L2a/L2b apply the same adapter to
everything, so for those the oracle column must be identical to the learned one.
That identity is a cheap correctness check on the whole setup.

Training is identical under every protocol, so each (level, seed) trains once
and is evaluated four times. The S0 record carries `protocol`,
`task_id_at_inference`, `class_masking` and `routing_mode` separately, and the
aggregator refuses to mix them (R3).

Usage:
    python experiments/s5_protocols.py --device cuda
    python experiments/s5_protocols.py --datasets cifar10 --levels L3_per_task --seeds 42
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

import s2_ladder  # noqa: E402
import s7_transfer  # noqa: E402
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402

LEVELS = [
    "L0_ncm",
    "L1_ridge",
    "L2a_shared_joint",
    "L2b_shared_seq",
    "L3_per_task",
    "L4_oracle",
]

# (protocol id, class_masking, oracle routing, human description)
CONDITIONS = [
    ("class_il", False, False, "Class-IL reference: all seen classes, learned router"),
    ("class_il_oracle", False, True, "task id for routing only, all seen classes"),
    ("task_il_mask", True, False, "class space restricted to the task, learned router"),
    ("task_il", True, True, "Task-IL: task id and class space restricted"),
]

DATASETS = {
    "cifar100": {
        "cache": "results/feature_cache/cifar100_vit_b16/feature_cache.pt",
        "downstream": "results/feature_cache/cifar10_vit_b16/feature_cache.pt",
    },
    "cifar10": {
        "cache": "results/feature_cache/cifar10_vit_b16/feature_cache.pt",
        "downstream": "results/feature_cache/cifar10_vit_b16/feature_cache.pt",
    },
}


def expert_entropy_mi(model, tasks, device, k: int = 1) -> dict:
    """MI(Expert;Task) and MI(Expert;Class) from the top-k routed experts.

    Built from a contingency table over every test sample of every task, so it
    answers "does the expert bank track tasks, or classes, or neither" without
    relying on the accuracy. Returns {} when the level has no router.
    """
    if model.router is None or not model.experts or model.spec.router == "oracle":
        return {}
    n_experts = len(model.experts)
    pairs = []
    with torch.no_grad():
        for task in tasks:
            feats, labels = task["splits"]["test"]
            feats = feats.to(device)
            ids, _ = model.router.top_k(feats, k=k)
            for j in range(ids.size(1)):
                pairs.append(
                    (
                        ids[:, j].cpu(),
                        torch.full_like(labels.cpu(), task["task_id"]),
                        labels.cpu(),
                    )
                )
    expert = torch.cat([p[0] for p in pairs])
    task_of = torch.cat([p[1] for p in pairs])
    label = torch.cat([p[2] for p in pairs])
    return {
        f"mi_expert_task@{k}": _normalized_mi(
            expert, task_of, n_experts, int(task_of.max()) + 1
        ),
        f"mi_expert_class@{k}": _normalized_mi(
            expert, label, n_experts, int(label.max()) + 1
        ),
        f"expert_entropy@{k}": _entropy(
            torch.bincount(expert, minlength=n_experts).float() / max(expert.numel(), 1)
        ),
    }


def _entropy(probs: torch.Tensor) -> float:
    p = probs[probs > 0]
    return float(-(p * p.log()).sum())


def _normalized_mi(a: torch.Tensor, b: torch.Tensor, n_a: int, n_b: int) -> float:
    table = torch.zeros(n_a, n_b)
    flat = a.long() * n_b + b.long()
    table.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
    total = table.sum()
    if total == 0:
        return float("nan")
    p = table / total
    pa = p.sum(dim=1, keepdim=True)
    pb = p.sum(dim=0, keepdim=True)
    denom = pa @ pb
    mask = p > 0
    mi = float((p[mask] * (p[mask] / denom[mask]).log()).sum())
    ha = _entropy(pa.flatten())
    hb = _entropy(pb.flatten())
    norm = min(ha, hb)
    return mi / norm if norm > 1e-9 else 0.0


def run_cell(
    dataset: str, level: str, paths: dict, downstream: dict, args, device
) -> list[dict]:
    """Train once for (dataset, level, seed) and evaluate under every protocol."""
    meta, tasks = s2_ladder.load_tasks(paths["cache"])
    dim = int(meta["feature_dim"])
    num_classes = sum(len(t["classes"]) for t in tasks)
    spec = s2_ladder.LEVELS_BY_NAME[level]

    raw = s7_transfer.transfer_suite(
        downstream["train"][0],
        downstream["train"][1],
        downstream["test"][0],
        downstream["test"][1],
        args.seed,
    )

    s2_ladder.set_seed(args.seed)
    model = s2_ladder.LadderModel(spec, dim, num_classes, args, device)
    n = len(tasks)
    fwt: list[float] = []
    if spec.joint:
        all_feats = torch.cat([t["splits"]["train"][0] for t in tasks], dim=0)
        all_labels = torch.cat([t["splits"]["train"][1] for t in tasks], dim=0)
        model.seen = sorted({c for t in tasks for c in t["classes"]})
        for i, task in enumerate(tasks):
            model.register_task(task, i)
        model.fit_task(None, 0, joint_data=(all_feats, all_labels))
        if spec.readout in s2_ladder.CLOSED_FORM_READOUTS:
            model.readout.fit(all_feats, all_labels, seen_classes=model.seen)
    else:
        for t, task in enumerate(tasks):
            if model.seen:
                _a, delta = s2_ladder.forward_transfer(model, task, num_classes, dim)
                fwt.append(delta)
            model.seen = sorted(set(model.seen) | set(task["classes"]))
            model.fit_task(task, t)
            model.register_task(task, t)

    with torch.no_grad():
        z_tr = model.represent(downstream["train"][0].to(device)).cpu()
        z_te = model.represent(downstream["test"][0].to(device)).cpu()
    transfer = s7_transfer.transfer_suite(
        z_tr, downstream["train"][1], z_te, downstream["test"][1], args.seed
    )
    mi = expert_entropy_mi(model, tasks, device)

    cells = []
    for protocol, masking, oracle, note in CONDITIONS:
        R = np.zeros((n, n), dtype=np.float32)
        # Training is protocol-independent, so only the evaluation changes.
        if spec.joint:
            for i in range(n):
                R[n - 1, i] = model.evaluate_task(
                    tasks[i], oracle=oracle, class_masking=masking
                )
        else:
            for t in range(n):
                for i in range(t + 1):
                    R[t, i] = model.evaluate_task(
                        tasks[i], oracle=oracle, class_masking=masking, task_id=i
                    )
        T = n - 1
        forget = [
            max(0.0, float(np.max(R[i : T + 1, i])) - float(R[T, i])) for i in range(T)
        ]
        cells.append(
            {
                "dataset": dataset,
                "level": level,
                "seed": args.seed,
                "protocol": protocol,
                "class_masking": masking,
                "routing_mode": "oracle" if oracle else "learned",
                "task_id_at_inference": bool(oracle),
                "increment_type": "class",
                "class_space": "task" if masking else "shared",
                "note": note,
                "num_tasks": n,
                "classes_per_task": len(tasks[0]["classes"]),
                "accuracy": float(np.mean(R[T, :])),
                "forgetting": float(np.mean(forget)) if forget else 0.0,
                "bwt": (
                    float(np.mean([R[T, i] - R[i, i] for i in range(T)])) if T else 0.0
                ),
                "fwt": float(np.mean(fwt)) if fwt else None,
                "acc_matrix": R.tolist(),
                "transfer": transfer,
                "delta_transfer": {k: transfer[k] - raw[k] for k in transfer},
                "raw_transfer": raw,
                "params": model.cost()["params"],
                "expert_count": 0 if not spec.expert else n,
                **mi,
            }
        )
    return cells


def aggregate(cells: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = {}
    for cell in cells:
        grouped.setdefault(
            (cell["dataset"], cell["level"], cell["protocol"]), []
        ).append(cell)

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

    table: dict[str, dict] = {}
    for (dataset, level, protocol), rows in grouped.items():
        table.setdefault(dataset, {}).setdefault(level, {})[protocol] = {
            "accuracy": st([r["accuracy"] for r in rows]),
            "forgetting": st([r["forgetting"] for r in rows]),
            "fwt": st([r["fwt"] for r in rows]),
            "transfer_full": st([r["transfer"]["full"] for r in rows]),
            "transfer_few1": st([r["transfer"]["few1"] for r in rows]),
            "mi_expert_task": st([r.get("mi_expert_task@1") for r in rows]),
            "mi_expert_class": st([r.get("mi_expert_class@1") for r in rows]),
            "seeds": [r["seed"] for r in rows],
        }

    # The two contributions of the protocol change, per level.
    deltas: dict[str, dict] = {}
    for dataset, levels in table.items():
        deltas[dataset] = {}
        for level, protocols in levels.items():

            def acc(protocol, protocols=protocols):
                entry = protocols.get(protocol, {}).get("accuracy")
                return entry["mean"] if entry else None

            base = acc("class_il")
            entry = {}
            if base is not None and acc("task_il") is not None:
                entry["task_il_minus_class_il"] = acc("task_il") - base
            if base is not None and acc("task_il_mask") is not None:
                entry["class_masking_effect"] = acc("task_il_mask") - base
            if base is not None and acc("class_il_oracle") is not None:
                entry["routing_effect"] = acc("class_il_oracle") - base
            deltas[dataset][level] = entry
    return {"matrix": table, "deltas": deltas}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="cifar100,cifar10")
    parser.add_argument("--levels", nargs="*", default=LEVELS)
    parser.add_argument("--seeds", default="42,1,2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument("--max_experts", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/s5")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    datasets = [d for d in args.datasets.split(",") if d]
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    study_path = os.path.join(args.out, "s5_protocol_study.json")

    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "rank": args.rank,
        "lambda_func": args.lambda_func,
        "levels": list(args.levels),
        "protocols": [c[0] for c in CONDITIONS],
    }
    cells: list[dict] = []
    done: set[tuple] = set()
    if os.path.exists(study_path) and not args.force:
        previous = json.load(open(study_path))
        if previous.get("recipe") == recipe:
            cells = [c for c in previous.get("cells", []) if c["dataset"] in datasets]
            done = {(c["dataset"], c["level"], c["seed"]) for c in cells}
            print(f"[S5] resuming: {len(done)} trained cells recorded", flush=True)
        else:
            print(
                "[S5] recorded study used a different recipe; starting fresh",
                flush=True,
            )

    def save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "s5_protocol_axis",
            "backbone": "vit_b_16+proj768",
            "protocols": [
                {
                    "id": c[0],
                    "class_masking": c[1],
                    "routing": "oracle" if c[2] else "learned",
                    "note": c[3],
                }
                for c in CONDITIONS
            ],
            "seeds": seeds,
            "recipe": recipe,
            "cells": cells,
            "aggregate": aggregate(cells) if cells else {},
            "contracts": {
                f"{c['dataset']}__{c['level']}__{c['protocol']}__seed{c['seed']}": _contract(
                    c
                )
                for c in cells
            },
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)

    for dataset in datasets:
        paths = DATASETS[dataset]
        if not os.path.exists(paths["cache"]):
            print(f"[S5] {dataset}: missing cache, skipping", flush=True)
            continue
        downstream = s7_transfer.load_downstream(paths["downstream"])
        for level in args.levels:
            for seed in seeds:
                if (dataset, level, seed) in done:
                    continue
                args.seed = seed
                produced = run_cell(dataset, level, paths, downstream, args, device)
                cells.extend(produced)
                save()
                line = "  ".join(
                    f"{c['protocol']}={c['accuracy'] * 100:5.2f}%" for c in produced
                )
                print(
                    f"[S5] {dataset:9s} {level:16s} seed={seed:<3d} {line}", flush=True
                )

    save()
    print(f"[S5] wrote {study_path}", flush=True)
    _print(aggregate(cells))


def _contract(cell: dict) -> dict:
    record = build_run_record(
        factors={
            "dataset": cell["dataset"],
            "protocol": cell["protocol"],
            "task_id_at_inference": cell["task_id_at_inference"],
            "class_masking": cell["class_masking"],
            "routing_mode": cell["routing_mode"],
            "increment_type": cell["increment_type"],
            "class_space": cell["class_space"],
            "num_tasks": cell["num_tasks"],
            "classes_per_task": cell["classes_per_task"],
            "seed": cell["seed"],
            "model_family": cell["level"],
            "backbone": "vit_b_16+proj768",
            "backbone_pretraining": "imagenet_frozen",
            "readout": "ncm" if cell["level"] == "L0_ncm" else "ridge",
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
            "cost": {"stored_bytes": 0, "total_params": cell["params"]},
            "generalization": {"transfer_accuracy": cell["transfer"]["full"]},
            "modular": {
                "expert_count": cell["expert_count"],
                "routing_entropy": None,
                "utilization": None,
            },
        },
        provenance={
            "command": "experiments/s5_protocols.py",
            "note": cell["note"],
            "mi_expert_task": cell.get("mi_expert_task@1"),
            "mi_expert_class": cell.get("mi_expert_class@1"),
            "delta_transfer": cell["delta_transfer"],
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


def _print(agg: dict) -> None:
    print("\n" + "=" * 96)
    print("S5 PROTOCOL AXIS (frozen ViT-B/16, CIFAR-scale, mean over seeds)")
    print("=" * 96)
    for dataset, levels in agg["matrix"].items():
        print(f"\n{dataset}")
        print(
            f"  {'level':16s} {'ClassIL':>9s} {'+oracle':>9s} {'mask':>9s} {'TaskIL':>9s} "
            f"{'route eff':>10s} {'mask eff':>10s} {'TaskIL-IL':>10s}"
        )
        for level, protocols in levels.items():

            def a(p, protocols=protocols):
                e = protocols.get(p, {}).get("accuracy")
                return e["mean"] if e else None

            def f(x):
                return f"{x * 100:8.2f}%" if x is not None else "       -"

            d = agg["deltas"][dataset][level]
            print(
                f"  {level:16s} {f(a('class_il'))} {f(a('class_il_oracle'))} "
                f"{f(a('task_il_mask'))} {f(a('task_il'))} "
                f"{f(d.get('routing_effect'))} {f(d.get('class_masking_effect'))} "
                f"{f(d.get('task_il_minus_class_il'))}"
            )


if __name__ == "__main__":
    main()
