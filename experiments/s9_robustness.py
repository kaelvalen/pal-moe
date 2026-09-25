"""
S9 - does the decomposition survive a distribution shift?

S8 closed the capacity question: more rank, more experts per sample and 16x the
router memory do not close the routing tax, and in the hard-routing regime the
whole expert bank is worth nothing over a training-free NCM. S9 asks whether
that decomposition is an artifact of the clean benchmark geometry:

    clean  ->  corruption / spurious shift  ->  routing tax, isolation
                                               realization, unseen transfer

Two controlled families, each varying exactly one thing:

    corruption   training stays clean, only the test-time input distribution
                 changes: gaussian noise / defocus blur / brightness at three
                 severities (a controlled CIFAR-C style family, not the
                 official CIFAR-100-C benchmark)
    spurious     a 6x6 corner cue is present during training on the first half
                 of the classes; at test it is correlated, absent, or flipped
                 onto the other half. Training on clean and testing on the same
                 conditions is the control arm, so the shortcut effect is the
                 *interaction* between training and test condition.

Both S6b constructions are run, because S6b showed the routing tax is a function
of the partition (13.98 `coherent` against 26.63 `dispersed`) and the point of
S9 is whether that decomposition survives a shift.

The metrics are the S5b/S6b/S8 ones - accuracy, `L4 - L3`, `L3 - L2b`,
`R_iso_ncm`, `recall@3`, assignment max-share and entropy - and *not* forgetting,
which S8 measured to be a floor effect in this ladder.

Shift caches are keyed by the canonical partition, so they are regrouped by
*label* onto each construction: a construction is a regrouping of class ids, and
selecting by label is order-independent and works for the train split too (whose
order comes from a permutation).

Everything is at the S8 operating point (rank 8, one prototype per class,
`top_k = 1`), so the clean cells must reproduce S8's seed-42 cells exactly.

Usage:
    python experiments/s9_robustness.py --device cuda
    python experiments/s9_robustness.py --families baseline --constructs dispersed
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
import s6b_difficulty  # noqa: E402
import s9_corruptions as shifts  # noqa: E402
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402

SOURCE_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"
CLEAN = "clean"
SPURIOUS_TRAIN = "spurious_train"
LEVELS = ["L0_ncm", "L1_ridge", "L2b_shared_seq", "L3_per_task", "L4_oracle"]
OPERATING = {"rank": 8, "protos": 1, "top_k": 1}


def corruption_conditions() -> list[str]:
    return [
        f"{name}_s{severity}"
        for name in shifts.CORRUPTIONS
        for severity in shifts.SEVERITIES
    ]


def spurious_conditions() -> list[str]:
    return [f"spurious_{mode}" for mode in shifts.SPURIOUS_MODES]


def default_grid(constructs, levels, seeds, families=None) -> list[dict]:
    """Every (construct, train, test, level, seed) cell."""
    cells: list[dict] = []

    def add(construct, train, test, family, level, seed):
        cells.append(
            {
                "construct": construct,
                "train_condition": train,
                "test_condition": test,
                "family": family,
                "level": level,
                "seed": int(seed),
            }
        )

    for construct in constructs:
        for level in levels:
            for seed in seeds:
                add(construct, CLEAN, CLEAN, "baseline", level, seed)
                for condition in corruption_conditions():
                    add(construct, CLEAN, condition, "corruption", level, seed)
                for condition in spurious_conditions():
                    add(construct, CLEAN, condition, "spurious_control", level, seed)
                add(
                    construct, SPURIOUS_TRAIN, CLEAN, "spurious_train_only", level, seed
                )
                for condition in spurious_conditions():
                    add(construct, SPURIOUS_TRAIN, condition, "spurious", level, seed)
    unique = {}
    for cell in cells:
        unique[_key(cell)] = cell
    out = list(unique.values())
    if families:
        wanted = set(families)
        out = [c for c in out if c["family"] in wanted]
    return out


def _key(cell: dict) -> tuple:
    return (
        cell["construct"],
        cell["train_condition"],
        cell["test_condition"],
        cell["level"],
        cell["seed"],
    )


# ---------------------------------------------------------------------------
# data: shift caches are canonical, regrouped by label onto each construction
# ---------------------------------------------------------------------------


def _concat(cache: dict, split: str):
    feats = torch.cat([row[split][0] for row in cache["tasks"].values()], dim=0)
    labels = torch.cat([row[split][1] for row in cache["tasks"].values()], dim=0)
    return feats, labels


def _regroup(feats: torch.Tensor, labels: torch.Tensor, classes) -> tuple:
    mask = torch.isin(labels, torch.tensor([int(c) for c in classes]))
    return feats[mask], labels[mask]


def load_condition(name: str, base_tasks: dict) -> dict:
    """Test tasks per construction for one test condition."""
    if name == CLEAN:
        return base_tasks
    path = shifts.cache_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing shift cache {path}")
    cache = shifts.load_shift_cache(path)
    feats, labels = _concat(cache, "test")
    out = {}
    for construct, tasks in base_tasks.items():
        regrouped = []
        for task in tasks:
            task_feats, task_labels = _regroup(feats, labels, task["classes"])
            regrouped.append(
                {
                    "task_id": int(task["task_id"]),
                    "classes": list(task["classes"]),
                    "splits": {
                        "train": task["splits"]["train"],
                        "val": task["splits"]["val"],
                        "test": (task_feats, task_labels),
                    },
                }
            )
        out[construct] = regrouped
    return out


def load_train_condition(name: str, base_tasks: dict) -> dict:
    """Train/val tasks per construction for one training condition."""
    if name == CLEAN:
        return base_tasks
    path = shifts.cache_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing shift cache {path}")
    cache = shifts.load_shift_cache(path)
    parts = {split: _concat(cache, split) for split in ("train", "val")}
    out = {}
    for construct, tasks in base_tasks.items():
        regrouped = []
        for task in tasks:
            splits = {}
            for split in ("train", "val"):
                feats, labels = parts[split]
                splits[split] = _regroup(feats, labels, task["classes"])
            splits["test"] = task["splits"]["test"]
            regrouped.append(
                {
                    "task_id": int(task["task_id"]),
                    "classes": list(task["classes"]),
                    "splits": splits,
                }
            )
        out[construct] = regrouped
    return out


def verify_regroup(base_tasks: dict, source_tasks: list[dict]) -> dict:
    """The regrouped clean test split must equal the construction's own split.

    `build_construction` regroups the canonical cache by class as well, so this
    checks the regrouping machinery against the code S6b/S8 already trust - and
    that every construction task still holds 100 test samples per class.
    """
    feats = torch.cat([t["splits"]["test"][0] for t in source_tasks], dim=0)
    labels = torch.cat([t["splits"]["test"][1] for t in source_tasks], dim=0)
    worst = 0.0
    counts_ok = True
    for tasks in base_tasks.values():
        for task in tasks:
            regrouped_feats, regrouped_labels = _regroup(feats, labels, task["classes"])
            own_feats, own_labels = task["splits"]["test"]
            if regrouped_feats.shape != own_feats.shape:
                counts_ok = False
                continue
            worst = max(worst, float((regrouped_feats - own_feats).abs().max()))
            per_class = [
                int((regrouped_labels == int(c)).sum()) for c in task["classes"]
            ]
            counts_ok = counts_ok and all(n == 100 for n in per_class)
    return {"max_abs_delta": worst, "counts_ok": counts_ok}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def train_model(level: str, train_tasks: list[dict], args, device):
    dim = int(train_tasks[0]["splits"]["train"][0].size(1))
    num_classes = sum(len(t["classes"]) for t in train_tasks)
    spec = s2_ladder.LEVELS_BY_NAME[level]
    cell_args = argparse.Namespace(
        rank=OPERATING["rank"],
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        lambda_func=args.lambda_func,
        max_experts=args.max_experts,
    )
    s2_ladder.set_seed(args.seed)
    model = s2_ladder.LadderModel(
        spec, dim, num_classes, cell_args, device, router_slots=num_classes
    )
    model.router_prototypes = OPERATING["protos"]
    model.top_k_experts = OPERATING["top_k"]
    for t, task in enumerate(train_tasks):
        model.seen = sorted(set(model.seen) | set(task["classes"]))
        model.fit_task(task, t)
        model.register_task(task, t)
    return model


@torch.no_grad()
def evaluate(model, level: str, test_tasks: list[dict]) -> dict:
    """The whole accuracy matrix is computed *after* training, as in S6b/S8.

    Interleaving the evaluation with training would silently change the
    protocol: `mask_unseen` restricts the head to the classes seen so far, so an
    evaluation taken at task t competes against 5(t+1) classes while the final
    one competes against 100.
    """
    spec = s2_ladder.LEVELS_BY_NAME[level]
    oracle = spec.router == "oracle"
    n = len(test_tasks)
    R = np.zeros((n, n), dtype=np.float32)
    for t in range(n):
        for i in range(t + 1):
            R[t, i] = model.evaluate_task(test_tasks[i], oracle=oracle, task_id=i)
    T = n - 1
    return {
        "accuracy": float(np.mean(R[T, :])),
        "retention": float(np.mean(R[T, :T])) if T > 0 else 0.0,
        "acc_matrix": R.tolist(),
        "routing": model.routing_stats(test_tasks, k=3),
        "resources": model.resources(),
    }


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def aggregate(cells: list[dict], constructs, levels) -> dict:
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

    def pick(**kw):
        return [c for c in cells if all(c[k] == v for k, v in kw.items())]

    out: dict[str, list] = {}
    all_tests = [CLEAN] + corruption_conditions() + spurious_conditions()
    for construct in constructs:
        conditions = []
        for train in (CLEAN, SPURIOUS_TRAIN):
            for test in all_tests:
                rows = pick(
                    construct=construct, train_condition=train, test_condition=test
                )
                if not rows:
                    continue
                by_level = {
                    level: st([r["accuracy"] for r in rows if r["level"] == level])
                    for level in levels
                }
                entry = {
                    "train_condition": train,
                    "test_condition": test,
                    "levels": by_level,
                    "recall_at_3": {
                        level: st(
                            [
                                r["routing"].get("task_recall_at_3")
                                for r in rows
                                if r["level"] == level
                            ]
                        )
                        for level in levels
                    },
                    "expert_max_share": {
                        level: st(
                            [
                                r["routing"].get("expert_max_share")
                                for r in rows
                                if r["level"] == level
                            ]
                        )
                        for level in levels
                    },
                    "expert_entropy": {
                        level: st(
                            [
                                r["routing"].get("expert_entropy")
                                for r in rows
                                if r["level"] == level
                            ]
                        )
                        for level in levels
                    },
                    "n": len(rows),
                }
                l0 = by_level.get("L0_ncm")
                l2b = by_level.get("L2b_shared_seq")
                l3 = by_level.get("L3_per_task")
                l4 = by_level.get("L4_oracle")
                if l3 and l4:
                    entry["routing_tax"] = l4["mean"] - l3["mean"]
                if l3 and l2b:
                    entry["isolation_realised"] = l3["mean"] - l2b["mean"]
                if l4 and l2b:
                    entry["isolation_available"] = l4["mean"] - l2b["mean"]
                if l3 and l0 and l4 and (l4["mean"] - l0["mean"]) > 1e-9:
                    entry["R_iso_ncm"] = (l3["mean"] - l0["mean"]) / (
                        l4["mean"] - l0["mean"]
                    )
                conditions.append(entry)
        out[construct] = conditions
    return {"constructs": out}


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def verify_against_s8(
    cells: list[dict], path="results/s8/s8_budget_study.json"
) -> dict:
    """The clean cells must reproduce S8's seed-42 operating point exactly.

    The first S9 attempt failed this guard and it was right to: it trained on
    the canonical CIFAR-100 partition while S8's stored cells are `dispersed`,
    which differs by 0.1 points (70.56 against 70.66). Training is
    bit-reproducible across processes - verified by hashing the model init and
    the per-task parameters in independent processes - so an exact comparison is
    the correct test, and the tolerance is only float noise.
    """
    if not os.path.exists(path):
        return {"checked": 0, "note": "S8 study not found"}
    reference = {
        (c["construct"], c["level"]): c["accuracy"]
        for c in json.load(open(path))["cells"]
        if c["rank"] == OPERATING["rank"]
        and c["protos"] == OPERATING["protos"]
        and c["top_k"] == OPERATING["top_k"]
        and c["seed"] == 42
    }
    here = {
        (c["construct"], c["level"]): c["accuracy"]
        for c in cells
        if c["train_condition"] == CLEAN
        and c["test_condition"] == CLEAN
        and c["seed"] == 42
    }
    matched = sorted(set(reference) & set(here))
    deltas = {f"{k[0]}/{k[1]}": abs(reference[k] - here[k]) for k in matched}
    worst = max(deltas.values()) if deltas else None
    return {
        "checked": len(matched),
        "max_abs_delta": worst,
        "tolerance": 1e-6,
        "within_tolerance": bool(worst is not None and worst <= 1e-6),
        "deltas": deltas,
        "note": "clean S9 cells vs S8's seed-42 operating point, per construct",
    }


def _contract(cell: dict) -> dict:
    spec = s2_ladder.LEVELS_BY_NAME[cell["level"]]
    resources = cell["resources"]
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
            "classes_per_task": 5,
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
                "forgetting": 0.0,
                "acc_matrix": cell["acc_matrix"],
            },
            "cost": {
                "stored_bytes": resources["memory_bytes"],
                "total_params": resources["total_params"],
            },
        },
        provenance={
            "command": "experiments/s9_robustness.py",
            "construct": cell["construct"],
            "family": cell["family"],
            "train_condition": cell["train_condition"],
            "test_condition": cell["test_condition"],
            "retention": cell["retention"],
            "routing": cell["routing"],
            "resources": resources,
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


def _print(agg: dict) -> None:
    def pct(stats):
        return f"{stats['mean'] * 100:6.2f}" if stats else "   n/a"

    def num(value):
        return f"{value:6.3f}" if value is not None else "   n/a"

    print("\n" + "=" * 108)
    print("S9 ROBUSTNESS (CIFAR-100, frozen ViT-B/16, Class-IL, seed 42)")
    print("=" * 108)
    for construct, conditions in agg["constructs"].items():
        print(f"\n########## {construct}")
        for entry in conditions:
            print(
                f"\n### train={entry['train_condition']:14s} "
                f"test={entry['test_condition']:22s} (n={entry['n']})"
            )
            print(f"  {'level':16s} {'acc':>7} {'recall@3':>9} {'max-share':>10}")
            for level, stats in entry["levels"].items():
                print(
                    f"  {level:16s} {pct(stats)} "
                    f"{num((entry['recall_at_3'][level] or {}).get('mean'))} "
                    f"{num((entry['expert_max_share'][level] or {}).get('mean'))}"
                )
            print(
                f"  -> tax={num(entry.get('routing_tax'))}  "
                f"realised={num(entry.get('isolation_realised'))}  "
                f"available={num(entry.get('isolation_available'))}  "
                f"R_iso_ncm={num(entry.get('R_iso_ncm'))}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=SOURCE_CACHE)
    parser.add_argument("--constructs", nargs="*", default=["dispersed", "coherent"])
    parser.add_argument("--levels", nargs="*", default=LEVELS)
    parser.add_argument("--seeds", default="42")
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="restrict to families: baseline corruption spurious_control "
        "spurious_train_only spurious",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument("--max_experts", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/s9")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    _, source_tasks = s2_ladder.load_tasks(args.cache)
    study_path = os.path.join(args.out, "s9_robustness_study.json")

    base_tasks = {
        construct: s6b_difficulty.build_construction(source_tasks, construct)
        for construct in args.constructs
    }
    regroup = verify_regroup(base_tasks, source_tasks)
    print(f"[S9] regroup guard: {regroup}", flush=True)
    if not regroup["counts_ok"] or regroup["max_abs_delta"] > 1e-3:
        raise SystemExit("regrouping by label does not reproduce the construction")

    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_func": args.lambda_func,
        "levels": list(args.levels),
        "constructs": list(args.constructs),
        "operating": OPERATING,
        "corruptions": shifts.CORRUPTIONS,
        "severities": shifts.SEVERITIES,
        "spurious_modes": shifts.SPURIOUS_MODES,
        "regroup_guard": regroup,
    }
    cells: list[dict] = []
    done: set[tuple] = set()
    if os.path.exists(study_path) and not args.force:
        previous = json.load(open(study_path))
        if previous.get("recipe") == recipe:
            cells = previous.get("cells", [])
            done = {_key(c) for c in cells}
            print(f"[S9] resuming: {len(done)} cells recorded", flush=True)

    def save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "s9_robustness",
            "backbone": "vit_b_16+proj768",
            "recipe": recipe,
            "seeds": seeds,
            "cells": cells,
            "aggregate": (
                aggregate(cells, args.constructs, args.levels) if cells else {}
            ),
            "contracts": {
                "__".join(str(part) for part in _key(c)): _contract(c) for c in cells
            },
            "s8_reproduction_guard": verify_against_s8(cells),
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)

    grid = default_grid(args.constructs, args.levels, seeds, args.families)
    print(f"[S9] grid: {len(grid)} cells, {len(done)} already done", flush=True)

    train_cache: dict[tuple, dict] = {}
    test_cache: dict[str, dict] = {}
    order: dict[tuple, list] = {}
    for cell in grid:
        order.setdefault(
            (cell["construct"], cell["train_condition"], cell["level"], cell["seed"]),
            [],
        ).append(cell)

    for (construct, train_condition, level, seed), group in order.items():
        pending = [c for c in group if _key(c) not in done]
        if not pending:
            continue
        cache_key = (construct, train_condition)
        if cache_key not in train_cache:
            train_cache[cache_key] = load_train_condition(train_condition, base_tasks)[
                construct
            ]
        args.seed = seed
        model = train_model(level, train_cache[cache_key], args, device)
        print(
            f"[S9] trained {construct:10s} {level:16s} on "
            f"{train_condition:14s} seed={seed} -> {len(pending)} tests",
            flush=True,
        )
        for cell in pending:
            condition = cell["test_condition"]
            if condition not in test_cache:
                test_cache[condition] = load_condition(condition, base_tasks)
            result = evaluate(model, level, test_cache[condition][construct])
            cells.append({**cell, **result, "num_tasks": len(train_cache[cache_key])})
            done.add(_key(cell))
            save()
            print(
                f"[S9]   test={condition:22s} "
                f"acc={result['accuracy'] * 100:6.2f}%  "
                f"recall@3={(result['routing'].get('task_recall_at_3') or float('nan')):.3f}  "
                f"share={(result['routing'].get('expert_max_share') or float('nan')):.3f}",
                flush=True,
            )

    save()
    print(f"[S9] wrote {study_path}", flush=True)
    payload = json.load(open(study_path))
    _print(payload["aggregate"])
    print(f"\n[S9 guard] clean cells vs S8: {payload['s8_reproduction_guard']}")


if __name__ == "__main__":
    main()
