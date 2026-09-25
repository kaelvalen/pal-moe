"""
S6 - order sensitivity: are the routing tax and the isolation benefit properties
of the *stream*, or of the particular order the stream happens to be in?

Both axes are free: a class order is a re-partition of an existing feature cache
(which classes land in which task, features untouched) and a task order is a
permutation of the resulting groups. No new extraction, so the axis costs only
the ladder runs.

    class_order_seed  -> which classes share a task
    task_order_seed   -> the sequence the tasks are visited in

A held-out task adds the forward-looking questions the earlier stages could not
ask: does the representation make an unseen task easy to learn, and does the
router hand it to one expert or scatter it? The naive "accuracy on the unseen
task" is structurally zero (its classes were never seen, so they are masked),
which is the same trap FWT fell into - so the measurement is a closed-form probe
on the representation plus the router's expert distribution for that task.

Reported: accuracy / forgetting / BWT / FWT per order, the routing tax per
order, the variance across orders, and the correlation between the routing tax
and two order-difficulty proxies:

    ncm_difficulty   how hard the order is for the training-free readout
    order_confusion  how confusable consecutive tasks are in the representation
                     (mean max cosine similarity between their class means)

Usage:
    python experiments/s6_order.py --device cuda
    python experiments/s6_order.py --levels L1_ridge L3_per_task L4_oracle --orders 3
"""

import argparse
import itertools
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
from pal_moe.arch import build_readout  # noqa: E402
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402

LEVELS = ["L0_ncm", "L1_ridge", "L2b_shared_seq", "L3_per_task", "L4_oracle"]
SOURCE_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"


# ---------------------------------------------------------------------------
# re-partitioning: the whole point of S6 being cheap
# ---------------------------------------------------------------------------


def pool_tasks(tasks: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    feats = torch.cat([t["splits"]["train"][0] for t in tasks], dim=0)
    labels = torch.cat([t["splits"]["train"][1] for t in tasks], dim=0)
    return feats, labels


def repartition(
    tasks: list[dict],
    class_order_seed: int,
    task_order_seed: int,
    holdout_task: int | None = None,
) -> tuple[list[dict], dict | None]:
    """Re-group the cached samples into new tasks and reorder them.

    `class_order_seed` permutes which classes share a task; `task_order_seed`
    permutes the sequence. Both act on the cached features only, so the input
    distribution is bit-identical across every configuration - which is what
    makes the order the single variable.
    """
    per_task = len(tasks[0]["classes"])
    n_tasks = len(tasks)
    all_classes = sorted({int(c) for t in tasks for c in t["classes"]})
    gen = torch.Generator().manual_seed(class_order_seed)
    permuted = [
        all_classes[i] for i in torch.randperm(len(all_classes), generator=gen).tolist()
    ]

    feats, labels = pool_tasks(tasks)
    test_feats, test_labels = (
        torch.cat([t["splits"]["test"][0] for t in tasks], dim=0),
        torch.cat([t["splits"]["test"][1] for t in tasks], dim=0),
    )

    groups = []
    for g in range(n_tasks):
        classes = permuted[g * per_task : (g + 1) * per_task]
        train_mask = torch.isin(labels, torch.tensor(classes))
        test_mask = torch.isin(test_labels, torch.tensor(classes))
        groups.append(
            {
                "classes": classes,
                "train": (feats[train_mask], labels[train_mask]),
                "test": (test_feats[test_mask], test_labels[test_mask]),
            }
        )

    hold = None
    if holdout_task is not None:
        hold = groups.pop(holdout_task % len(groups))

    gen = torch.Generator().manual_seed(task_order_seed)
    order = torch.randperm(len(groups), generator=gen).tolist()
    groups = [groups[i] for i in order]

    rebuilt = []
    for index, group in enumerate(groups):
        rebuilt.append(
            {
                "task_id": index,
                "classes": group["classes"],
                "splits": {
                    "train": group["train"],
                    "val": group["test"],
                    "test": group["test"],
                },
            }
        )
    return rebuilt, hold


def order_confusion(tasks: list[dict]) -> float:
    """Mean max cosine similarity between class means of consecutive tasks.

    A representation-level difficulty proxy: how easy it is to confuse the next
    task with the current one.
    """
    means = []
    for task in tasks:
        feats, labels = task["splits"]["train"]
        per_class = []
        for c in task["classes"]:
            mask = labels == c
            if bool(mask.any()):
                per_class.append(feats[mask].mean(dim=0))
        if per_class:  # a re-partitioned cache always has samples per class
            means.append(torch.stack(per_class))
    sims = []
    for a, b in zip(means, means[1:]):
        sims.append(
            float(
                (
                    torch.nn.functional.normalize(a, dim=-1)
                    @ torch.nn.functional.normalize(b, dim=-1).t()
                ).max()
            )
        )
    return float(np.mean(sims)) if sims else float("nan")


# ---------------------------------------------------------------------------
# one configuration
# ---------------------------------------------------------------------------


def run_configuration(
    level: str,
    tasks: list[dict],
    holdout: dict | None,
    class_order_seed: int,
    task_order_seed: int,
    args,
    device,
) -> dict:
    dim = int(tasks[0]["splits"]["train"][0].size(1))
    num_classes = len({int(c) for t in tasks for c in t["classes"]}) + (
        len(holdout["classes"]) if holdout else 0
    )
    spec = s2_ladder.LEVELS_BY_NAME[level]

    s2_ladder.set_seed(args.seed)
    model = s2_ladder.LadderModel(spec, dim, num_classes, args, device)
    n = len(tasks)
    fwt: list[float] = []
    for t, task in enumerate(tasks):
        if model.seen:
            _a, delta = s2_ladder.forward_transfer(model, task, num_classes, dim)
            fwt.append(delta)
        model.seen = sorted(set(model.seen) | set(task["classes"]))
        model.fit_task(task, t)
        model.register_task(task, t)

    oracle = spec.router == "oracle"
    R = np.zeros((n, n), dtype=np.float32)
    for t in range(n):
        for i in range(t + 1):
            R[t, i] = model.evaluate_task(tasks[i], oracle=oracle, task_id=i)
    T = n - 1
    forget = [
        max(0.0, float(np.max(R[i : T + 1, i])) - float(R[T, i])) for i in range(T)
    ]

    holdout_metrics = {}
    if holdout is not None:
        # Accuracy on the unseen task is structurally zero (its classes are
        # masked), so measure the representation and the routing instead.
        # The probe runs on CPU: the representation is already moved there for
        # the transfer suite, and a closed-form ridge has no reason to hold a
        # device context.
        probe = build_readout("ridge", dim=dim, num_classes=num_classes)
        probe.fit(
            model.represent(holdout["train_x"].to(device)).cpu(),
            holdout["train_y"],
            seen_classes=sorted({int(c) for c in holdout["classes"]}),
        )
        holdout_metrics["unseen_task_transfer"] = float(
            (
                probe.predict(
                    model.represent(holdout["test_x"].to(device)).cpu()
                ).argmax(-1)
                == holdout["test_y"]
            )
            .float()
            .mean()
        )
        if model.router is not None and model.experts:
            with torch.no_grad():
                ids, _ = model.router.top_k(holdout["test_x"].to(device), k=1)
            share = torch.bincount(
                ids.flatten(), minlength=len(model.experts)
            ).float() / max(ids.numel(), 1)
            holdout_metrics["unseen_task_expert_max_share"] = float(share.max())
            holdout_metrics["unseen_task_expert_entropy"] = float(
                s5_protocols_entropy(share)
            )

    return {
        "level": level,
        "seed": args.seed,
        "class_order_seed": class_order_seed,
        "task_order_seed": task_order_seed,
        "accuracy": float(np.mean(R[T, :])),
        "forgetting": float(np.mean(forget)) if forget else 0.0,
        "bwt": float(np.mean([R[T, i] - R[i, i] for i in range(T)])) if T else 0.0,
        "fwt": float(np.mean(fwt)) if fwt else None,
        "acc_matrix": R.tolist(),
        "order_confusion": order_confusion(tasks),
        "params": model.cost()["params"],
        **holdout_metrics,
    }


def s5_protocols_entropy(probs: torch.Tensor) -> float:
    p = probs[probs > 0]
    return float(-(p * p.log()).sum())


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=SOURCE_CACHE)
    parser.add_argument("--levels", nargs="*", default=LEVELS)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--class_order_seeds", default="0,1,2")
    parser.add_argument("--task_order_seeds", default="0,1,2")
    parser.add_argument("--holdout_task", type=int, default=0, help="-1 disables")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument("--max_experts", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/s6")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    _, tasks = s2_ladder.load_tasks(args.cache)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    class_seeds = [int(s) for s in args.class_order_seeds.replace(",", " ").split()]
    task_seeds = [int(s) for s in args.task_order_seeds.replace(",", " ").split()]
    study_path = os.path.join(args.out, "s6_order_study.json")

    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "rank": args.rank,
        "levels": list(args.levels),
        "class_order_seeds": class_seeds,
        "task_order_seeds": task_seeds,
        "holdout_task": args.holdout_task,
    }
    cells: list[dict] = []
    done: set[tuple] = set()
    if os.path.exists(study_path) and not args.force:
        previous = json.load(open(study_path))
        if previous.get("recipe") == recipe:
            cells = previous.get("cells", [])
            done = {
                (c["level"], c["seed"], c["class_order_seed"], c["task_order_seed"])
                for c in cells
            }
            print(f"[S6] resuming: {len(done)} cells recorded", flush=True)
        else:
            print("[S6] different recipe, starting fresh", flush=True)

    def save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "s6_order_sensitivity",
            "backbone": "vit_b_16+proj768",
            "recipe": recipe,
            "seeds": seeds,
            "cells": cells,
            "aggregate": aggregate(cells) if cells else {},
            "contracts": {
                f"{c['level']}__cls{c['class_order_seed']}__tsk{c['task_order_seed']}__seed{c['seed']}": _contract(
                    c
                )
                for c in cells
            },
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)

    for class_seed, task_seed in itertools.product(class_seeds, task_seeds):
        rebuilt, holdout_raw = repartition(
            tasks,
            class_seed,
            task_seed,
            None if args.holdout_task < 0 else args.holdout_task,
        )
        holdout = None
        if holdout_raw is not None:
            holdout = {
                "classes": holdout_raw["classes"],
                "train_x": holdout_raw["train"][0],
                "train_y": holdout_raw["train"][1],
                "test_x": holdout_raw["test"][0],
                "test_y": holdout_raw["test"][1],
            }
        for level in args.levels:
            for seed in seeds:
                if (level, seed, class_seed, task_seed) in done:
                    continue
                args.seed = seed
                cell = run_configuration(
                    level, rebuilt, holdout, class_seed, task_seed, args, device
                )
                cells.append(cell)
                save()
                print(
                    f"[S6] {level:16s} cls={class_seed} tsk={task_seed} seed={seed} "
                    f"acc={cell['accuracy'] * 100:5.2f}% F={cell['forgetting'] * 100:5.2f}% "
                    f"conf={cell['order_confusion']:.3f}"
                    + (
                        f" unseen={cell['unseen_task_transfer'] * 100:5.2f}%"
                        if "unseen_task_transfer" in cell
                        else ""
                    ),
                    flush=True,
                )

    save()
    print(f"[S6] wrote {study_path}", flush=True)
    _print(aggregate(cells))


def aggregate(cells: list[dict]) -> dict:
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

    by_level: dict[str, list[dict]] = {}
    for cell in cells:
        by_level.setdefault(cell["level"], []).append(cell)

    levels = {
        level: {
            "accuracy": st([c["accuracy"] for c in rows]),
            "forgetting": st([c["forgetting"] for c in rows]),
            "order_confusion": st([c["order_confusion"] for c in rows]),
            "unseen_task_transfer": st([c.get("unseen_task_transfer") for c in rows]),
            "unseen_task_expert_max_share": st(
                [c.get("unseen_task_expert_max_share") for c in rows]
            ),
            "unseen_task_expert_entropy": st(
                [c.get("unseen_task_expert_entropy") for c in rows]
            ),
            "n_configurations": len(rows),
        }
        for level, rows in by_level.items()
    }

    def acc(level, class_seed, task_seed):
        rows = [
            c
            for c in cells
            if c["class_order_seed"] == class_seed
            and c["task_order_seed"] == task_seed
            and c["level"] == level
        ]
        return rows[0]["accuracy"] if rows else None

    # Routing tax and the isolation decomposition per order configuration.
    per_order: list[dict] = []
    configs = sorted({(c["class_order_seed"], c["task_order_seed"]) for c in cells})
    for class_seed, task_seed in configs:
        l0 = acc("L0_ncm", class_seed, task_seed)
        l1 = acc("L1_ridge", class_seed, task_seed)
        l2b = acc("L2b_shared_seq", class_seed, task_seed)
        l3 = acc("L3_per_task", class_seed, task_seed)
        l4 = acc("L4_oracle", class_seed, task_seed)
        conf = [
            c["order_confusion"]
            for c in cells
            if c["class_order_seed"] == class_seed and c["task_order_seed"] == task_seed
        ]
        per_order.append(
            {
                "class_order_seed": class_seed,
                "task_order_seed": task_seed,
                "L0_ncm": l0,
                "L1_ridge": l1,
                "L2b_shared_seq": l2b,
                "L3_per_task": l3,
                "L4_oracle": l4,
                # Difficulty for the training-free readout: higher = harder.
                "ncm_difficulty": (1.0 - l0) if l0 is not None else None,
                # The routing tax is the L4 - L3 quantity S4 measured.
                "routing_tax": (
                    (l4 - l3) if (l3 is not None and l4 is not None) else None
                ),
                # Isolation splits into what the learned router realises and
                # what is available under oracle routing; S5b showed the two
                # differ by a factor of three (class-IL 35%, domain-IL 15%).
                "isolation_realised": (
                    (l3 - l2b) if (l2b is not None and l3 is not None) else None
                ),
                "isolation_available": (
                    (l4 - l2b) if (l2b is not None and l4 is not None) else None
                ),
                "isolation_vs_ridge": (
                    (l3 - l1) if (l1 is not None and l3 is not None) else None
                ),
                "order_confusion": conf[0] if conf else None,
            }
        )

    def correlate(key_a: str, key_b: str) -> float | None:
        pairs = [
            (row[key_a], row[key_b])
            for row in per_order
            if row.get(key_a) is not None and row.get(key_b) is not None
        ]
        if len(pairs) < 3:
            return None
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        ma, mb = statistics.mean(a), statistics.mean(b)
        num = sum((x - ma) * (y - mb) for x, y in pairs)
        den = (
            sum((x - ma) ** 2 for x in a) ** 0.5 * sum((y - mb) ** 2 for y in b) ** 0.5
        )
        return num / den if den > 1e-12 else None

    # Easy / hard split on the median difficulty, so the answer is a shape
    # rather than a single coefficient: does the tax appear in every order
    # geometry, or only in the hard ones?
    groups: dict[str, dict] = {}
    scored = [row for row in per_order if row.get("ncm_difficulty") is not None]
    if len(scored) >= 4:
        median = statistics.median(row["ncm_difficulty"] for row in scored)
        for label, subset in (
            ("easy", [r for r in scored if r["ncm_difficulty"] <= median]),
            ("hard", [r for r in scored if r["ncm_difficulty"] > median]),
        ):
            if not subset:
                continue
            groups[label] = {
                "n_configurations": len(subset),
                "ncm_difficulty": statistics.mean(r["ncm_difficulty"] for r in subset),
                "routing_tax": statistics.mean(
                    r["routing_tax"] for r in subset if r["routing_tax"] is not None
                ),
                "isolation_realised": statistics.mean(
                    r["isolation_realised"]
                    for r in subset
                    if r["isolation_realised"] is not None
                ),
                "isolation_available": statistics.mean(
                    r["isolation_available"]
                    for r in subset
                    if r["isolation_available"] is not None
                ),
            }

    # Concentration and uncertainty are kept apart from transferability: a
    # router that always picks one expert is *confident*, and can still be
    # confidently wrong for a task it has never seen.
    unseen = {
        "max_share": levels.get("L3_per_task", {}).get("unseen_task_expert_max_share"),
        "entropy": levels.get("L3_per_task", {}).get("unseen_task_expert_entropy"),
        "transferability": levels.get("L3_per_task", {}).get("unseen_task_transfer"),
    }
    for entry in levels.values():
        share = entry.get("unseen_task_expert_max_share")
        if share and share["mean"] >= 0.5:
            entry["unseen_assignment_reading"] = "concentrated"
        elif share:
            entry["unseen_assignment_reading"] = "spread"
        else:
            entry["unseen_assignment_reading"] = None

    return {
        "levels": levels,
        "per_order": per_order,
        "difficulty_groups": groups,
        "unseen_assignment": unseen,
        "correlations": {
            "routing_tax_vs_order_confusion": correlate(
                "routing_tax", "order_confusion"
            ),
            "routing_tax_vs_ncm_difficulty": correlate("routing_tax", "ncm_difficulty"),
            "isolation_realised_vs_ncm_difficulty": correlate(
                "isolation_realised", "ncm_difficulty"
            ),
            "isolation_available_vs_ncm_difficulty": correlate(
                "isolation_available", "ncm_difficulty"
            ),
        },
    }


def _contract(cell: dict) -> dict:
    record = build_run_record(
        factors={
            "dataset": "cifar100",
            "protocol": "class_il",
            "task_id_at_inference": cell["level"] == "L4_oracle",
            "class_masking": False,
            "routing_mode": "oracle" if cell["level"] == "L4_oracle" else "learned",
            "increment_type": "class",
            "class_space": "shared",
            "seed": cell["seed"],
            "task_order_seed": cell["task_order_seed"],
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
            "generalization": {
                "unseen_task_accuracy": cell.get("unseen_task_transfer"),
            },
        },
        provenance={
            "command": "experiments/s6_order.py",
            "class_order_seed": cell["class_order_seed"],
            "order_confusion": cell["order_confusion"],
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


def _print(agg: dict) -> None:
    print("\n" + "=" * 100)
    print("S6 ORDER SENSITIVITY (CIFAR-100, frozen ViT-B/16)")
    print("=" * 100)
    print(
        f"{'level':16s} {'acc':>9s} {'std':>7s} {'forget':>9s} {'confusion':>10s} "
        f"{'unseen transfer':>16s} {'share':>7s} {'reading':>12s}"
    )
    for level, entry in agg["levels"].items():

        def f(x, w=8):
            return f"{x['mean'] * 100:{w}.2f}%" if x else " " * (w - 1) + "-"

        share = entry.get("unseen_task_expert_max_share")
        print(
            f"{level:16s} {f(entry['accuracy'], 8)} "
            f"{(entry['accuracy'] or {}).get('std', 0) * 100:6.2f}% "
            f"{f(entry['forgetting'], 8)} "
            f"{(entry['order_confusion'] or {}).get('mean', float('nan')):9.3f} "
            f"{f(entry['unseen_task_transfer'], 15)} "
            f"{(share['mean'] if share else float('nan')):7.3f} "
            f"{(entry.get('unseen_assignment_reading') or '-'):>12s}"
        )
    print("\nper order configuration:")
    print(
        f"  {'cls':>3s} {'tsk':>3s} {'L0':>7s} {'L1':>7s} {'L2b':>7s} {'L3':>7s} {'L4':>7s} "
        f"{'tax':>7s} {'iso(real)':>10s} {'iso(avail)':>11s} {'confus':>7s}"
    )
    for row in agg["per_order"]:

        def g(x, w=7):
            return f"{x * 100:{w}.2f}%" if x is not None else " " * (w - 1) + "-"

        print(
            f"  {row['class_order_seed']:3d} {row['task_order_seed']:3d} "
            f"{g(row['L0_ncm'])} {g(row['L1_ridge'])} {g(row['L2b_shared_seq'])} "
            f"{g(row['L3_per_task'])} {g(row['L4_oracle'])} {g(row['routing_tax'])} "
            f"{g(row['isolation_realised'], 10)} {g(row['isolation_available'], 11)} "
            f"{(row['order_confusion'] if row['order_confusion'] is not None else float('nan')):7.3f}"
        )
    print("\ndifficulty groups (median split on the training-free readout):")
    for label, entry in agg.get("difficulty_groups", {}).items():
        print(
            f"  {label:5s} n={entry['n_configurations']}  "
            f"difficulty={entry['ncm_difficulty']:.3f}  "
            f"tax={entry['routing_tax'] * 100:+.2f}%  "
            f"iso realised={entry['isolation_realised'] * 100:+.2f}%  "
            f"iso available={entry['isolation_available'] * 100:+.2f}%"
        )
    print("\nunseen-task assignment (L3): concentration is not correctness")
    ua = agg.get("unseen_assignment", {})
    share, transfer, entropy = (
        ua.get("max_share"),
        ua.get("transferability"),
        ua.get("entropy"),
    )
    parts = []
    if share:
        parts.append(f"max expert share={share['mean']:.3f}")
    if entropy:
        parts.append(f"expert entropy={entropy['mean']:.3f}")
    if transfer:
        parts.append(f"transferability={transfer['mean'] * 100:.2f}%")
    print("  " + " | ".join(parts) if parts else "  n/a")
    print("\ncorrelations:")
    for key, value in agg["correlations"].items():
        print(f"  {key}: {value:+.3f}" if value is not None else f"  {key}: n/a")


if __name__ == "__main__":
    main()
