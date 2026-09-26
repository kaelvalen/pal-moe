"""
S6b - the designed difficulty contrast S6 could not sample.

S6 varied the order by random permutation and found the routing tax unchanged,
but its own O3 admits why that is a weak test: a random permutation of CIFAR-100
classes produces tasks of nearly equal difficulty (0.289 vs 0.299), so there was
nothing to correlate against. This stage *designs* the contrast instead of
sampling it.

    coherent    one task per CIFAR-100 coarse label (20 superclasses x 5 classes)
                high intra-task similarity, low cross-task similarity
    dispersed   round-robin the classes so every task draws one class from each
                of five well-separated superclasses
                low intra-task similarity, high cross-task similarity

Both are built from the same cached features by regrouping class ids, so the
representation is bit-identical and the construction is the only variable.

The difficulty proxy is a global property of the construction rather than an
adjacency one, because a designed partition changes the whole geometry:

    intra_task_similarity   mean max cosine between distinct class means in a task
    cross_task_similarity   mean max cosine between class means of different tasks
    separability            intra - cross   (higher = easier to keep tasks apart)

Reported per construction: accuracy, the routing tax (`L4 - L3`), the isolation
decomposition (`L3 - L2b` realised, `L4 - L2b` available) and their relation to
separability.

Usage:
    python experiments/s6b_difficulty.py --device cuda
    python experiments/s6b_difficulty.py --constructs coherent --seeds 42
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
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402

LEVELS = ["L0_ncm", "L1_ridge", "L2b_shared_seq", "L3_per_task", "L4_oracle"]
SOURCE_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"

# Moved to `pal_moe.core.constructions` in the v3 restructure (phase 1); re-exported.
from pal_moe.core.constructions import (  # noqa: E402,F401
    args_data_dir,
    build_construction,
    separability,
    superclass_of,
)


def run_configuration(level: str, tasks: list[dict], args, device) -> dict:
    dim = int(tasks[0]["splits"]["train"][0].size(1))
    num_classes = sum(len(t["classes"]) for t in tasks)
    spec = s2_ladder.LEVELS_BY_NAME[level]

    s2_ladder.set_seed(args.seed)
    model = s2_ladder.LadderModel(spec, dim, num_classes, args, device)
    n = len(tasks)
    for t, task in enumerate(tasks):
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
    return {
        "level": level,
        "seed": args.seed,
        "num_tasks": n,
        "classes_per_task": len(tasks[0]["classes"]),
        "accuracy": float(np.mean(R[T, :])),
        "forgetting": float(np.mean(forget)) if forget else 0.0,
        "acc_matrix": R.tolist(),
        **separability(tasks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=SOURCE_CACHE)
    parser.add_argument("--constructs", nargs="*", default=["coherent", "dispersed"])
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
    parser.add_argument("--out", default="results/s6b")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    _, source = s2_ladder.load_tasks(args.cache)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    study_path = os.path.join(args.out, "s6b_difficulty_study.json")

    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "rank": args.rank,
        "lambda_func": args.lambda_func,
        "levels": list(args.levels),
        "constructs": list(args.constructs),
    }
    cells: list[dict] = []
    done: set[tuple] = set()
    if os.path.exists(study_path) and not args.force:
        previous = json.load(open(study_path))
        if previous.get("recipe") == recipe:
            cells = previous.get("cells", [])
            done = {(c["construct"], c["level"], c["seed"]) for c in cells}
            print(f"[S6b] resuming: {len(done)} cells recorded", flush=True)

    def save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "s6b_designed_difficulty",
            "backbone": "vit_b_16+proj768",
            "recipe": recipe,
            "seeds": seeds,
            "cells": cells,
            "aggregate": aggregate(cells) if cells else {},
            "contracts": {
                f"{c['construct']}__{c['level']}__seed{c['seed']}": _contract(c)
                for c in cells
            },
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)

    for construct in args.constructs:
        tasks = build_construction(source, construct)
        sep = separability(tasks)
        print(
            f"[S6b] {construct}: intra={sep['intra_task_similarity']:.3f} "
            f"cross={sep['cross_task_similarity']:.3f} "
            f"sep={sep['separability']:+.3f}",
            flush=True,
        )
        for level in args.levels:
            for seed in seeds:
                if (construct, level, seed) in done:
                    continue
                args.seed = seed
                cell = run_configuration(level, tasks, args, device)
                cell["construct"] = construct
                cells.append(cell)
                save()
                print(
                    f"[S6b] {construct:10s} {level:16s} seed={seed:<3d} "
                    f"acc={cell['accuracy'] * 100:6.2f}%",
                    flush=True,
                )

    save()
    print(f"[S6b] wrote {study_path}", flush=True)
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

    out: dict[str, dict] = {}
    for construct in sorted({c["construct"] for c in cells}):
        rows = [c for c in cells if c["construct"] == construct]
        levels: dict[str, dict] = {}
        for level in sorted({r["level"] for r in rows}):
            sub = [r for r in rows if r["level"] == level]
            levels[level] = {
                "accuracy": st([r["accuracy"] for r in sub]),
                "forgetting": st([r["forgetting"] for r in sub]),
            }
        out[construct] = {
            "levels": levels,
            "intra_task_similarity": st([r["intra_task_similarity"] for r in rows]),
            "cross_task_similarity": st([r["cross_task_similarity"] for r in rows]),
            "separability": st([r["separability"] for r in rows]),
            "routing_tax": _diff(levels, "L4_oracle", "L3_per_task"),
            "isolation_realised": _diff(levels, "L3_per_task", "L2b_shared_seq"),
            "isolation_available": _diff(levels, "L4_oracle", "L2b_shared_seq"),
            "n_configurations": len({(r["level"], r["seed"]) for r in rows}),
        }
    return out


def _diff(levels: dict, a: str, b: str) -> float | None:
    ea = levels.get(a, {}).get("accuracy")
    eb = levels.get(b, {}).get("accuracy")
    if not ea or not eb:
        return None
    return ea["mean"] - eb["mean"]


def _contract(cell: dict) -> dict:
    spec = s2_ladder.LEVELS_BY_NAME[cell["level"]]
    record = build_run_record(
        factors={
            "dataset": "cifar100",
            "protocol": "class_il",
            "task_id_at_inference": cell["level"] == "L4_oracle",
            "class_masking": False,
            "routing_mode": "oracle" if cell["level"] == "L4_oracle" else "learned",
            "increment_type": "class",
            "class_space": "task",
            "num_tasks": cell["num_tasks"],
            "classes_per_task": cell["classes_per_task"],
            "seed": cell["seed"],
            "task_order_seed": None,
            "model_family": cell["level"],
            "backbone": "vit_b_16+proj768",
            "backbone_pretraining": "imagenet_frozen",
            "readout": spec.readout,
            "expert": spec.expert or "none",
        },
        metrics={
            "learning": {
                "accuracy": cell["accuracy"],
                "forgetting": cell["forgetting"],
                "acc_matrix": cell["acc_matrix"],
            },
            "cost": {"stored_bytes": 0, "total_params": 0},
        },
        provenance={
            "command": "experiments/s6b_difficulty.py",
            "construct": cell["construct"],
            "separability": cell["separability"],
            "cross_task_similarity": cell["cross_task_similarity"],
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


def _print(agg: dict) -> None:
    print("\n" + "=" * 88)
    print("S6b DESIGNED DIFFICULTY (CIFAR-100, frozen ViT-B/16)")
    print("=" * 88)
    for construct, entry in agg.items():
        print(
            f"\n{construct}: intra={entry['intra_task_similarity']['mean']:.3f} "
            f"cross={entry['cross_task_similarity']['mean']:.3f} "
            f"separability={entry['separability']['mean']:+.3f}"
        )
        for level, stats in entry["levels"].items():
            acc = stats["accuracy"]
            print(
                f"  {level:16s} acc={acc['mean'] * 100:6.2f}% "
                f"(+-{acc['std'] * 100:.2f})"
            )

        def g(x):
            return f"{x * 100:+.2f}%" if x is not None else "n/a"

        print(
            f"  -> routing tax = {g(entry['routing_tax'])}   "
            f"isolation realised = {g(entry['isolation_realised'])}   "
            f"available = {g(entry['isolation_available'])}"
        )


if __name__ == "__main__":
    main()
