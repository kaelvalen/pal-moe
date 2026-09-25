"""
S8 - what does a budget buy, and in which routing regime?

S6b established that the routing tax is not a constant of the method but a
function of the task geometry (`intra_task_similarity` drives the oracle,
`cross_task_similarity` drives the tax), spanning 13.98 to 26.63 over two
designed partitions of the same data. A single accuracy-per-byte curve would
average over a factor of two in the quantity the budget is supposed to buy, so
S8 sweeps the budget in *both* regimes and reports the mechanism, not just the
accuracy:

    coherent    easy routing: does extra capacity get realised?
    dispersed   hard routing: how much of the budget does routing strand?

Three resource axes, one variable at a time, mapped explicitly to the mechanism
each one is supposed to buy:

    parameter   adapter rank            -> expert capacity
    memory      prototypes per class    -> routing resolution
    active      experts per sample      -> inference compute

The ladder levels stay the S2-S6 ones (L0, L1, L2b, L3, L4). L4 is an oracle
upper bound, not a budget method: it is measured *at every rank* so that the
comparison `L3 -> L4` at a fixed budget asks whether more capacity closes the
routing gap or whether the gap is routing realization and not capacity.

Primary mechanism metric, per regime and per budget point:

    R_iso = (L3 - L2b) / (L4 - L2b)      realised / available isolation

Usage:
    python experiments/s8_budget.py --device cuda
    python experiments/s8_budget.py --constructs coherent --ranks 2 8 32 --seeds 42
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
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402

SOURCE_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"
DEFAULT_RANKS = [2, 8, 32, 128]
DEFAULT_PROTOTYPES = [1, 4, 16]
DEFAULT_TOPK = [1, 2, 4]
EXPERT_LEVELS = ("L2b_shared_seq", "L3_per_task")
OPERATING_RANK = 8


def default_grid(
    constructs, ranks, prototypes, topks, seeds, sweep_seeds
) -> list[dict]:
    """One variable at a time: references, then the three axes.

    The parameter axis sweeps `L2b` and `L3` at every rank (the same budget
    spent on a shared adapter against a bank of per-task experts) and `L4` at
    every rank too, so the question "does more capacity close the routing gap"
    is asked at a fixed budget rather than across budgets.

    Reference cells (the S2-S6 operating point, rank 8) carry the full seed set
    and must reproduce S6b; the extra budget points are one-seed sweep points,
    because the sweep is read as a curve shape and S6b already supplies the
    three-seed values at the operating point.
    """
    out: list[dict] = []

    def add(construct, level, rank, protos, top_k, seed):
        out.append(
            {
                "construct": construct,
                "level": level,
                "rank": int(rank),
                "protos": int(protos),
                "top_k": int(top_k),
                "seed": int(seed),
            }
        )

    for construct in constructs:
        # references at the operating point: full seed set
        for level in (
            "L0_ncm",
            "L1_ridge",
            "L2b_shared_seq",
            "L3_per_task",
            "L4_oracle",
        ):
            for seed in seeds:
                add(construct, level, OPERATING_RANK, 1, 1, seed)
        # parameter axis: L2b/L3 at every rank, L4 at every rank
        for rank in ranks:
            if rank == OPERATING_RANK:
                continue
            for level in EXPERT_LEVELS + ("L4_oracle",):
                for seed in sweep_seeds:
                    add(construct, level, rank, 1, 1, seed)
        # memory axis at the operating point
        for protos in prototypes:
            if protos == 1:
                continue
            for seed in sweep_seeds:
                add(construct, "L3_per_task", OPERATING_RANK, protos, 1, seed)
        # active axis at the operating point
        for top_k in topks:
            if top_k == 1:
                continue
            for seed in sweep_seeds:
                add(construct, "L3_per_task", OPERATING_RANK, 1, top_k, seed)

    unique: dict[tuple, dict] = {}
    for cell in out:
        unique[_key(cell)] = cell
    return list(unique.values())


def _key(cell: dict) -> tuple:
    return (
        cell["construct"],
        cell["level"],
        cell["rank"],
        cell["protos"],
        cell["top_k"],
        cell["seed"],
    )


def run_cell(cell: dict, tasks: list[dict], args, device) -> dict:
    dim = int(tasks[0]["splits"]["train"][0].size(1))
    num_classes = sum(len(t["classes"]) for t in tasks)
    spec = s2_ladder.LEVELS_BY_NAME[cell["level"]]

    cell_args = argparse.Namespace(
        rank=cell["rank"],
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=cell["seed"],
        lambda_func=args.lambda_func,
        max_experts=args.max_experts,
    )
    s2_ladder.set_seed(cell["seed"])
    model = s2_ladder.LadderModel(
        spec, dim, num_classes, cell_args, device, router_slots=num_classes
    )
    model.router_prototypes = cell["protos"]
    model.top_k_experts = cell["top_k"]

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
    # Retention, not forgetting, is what the freeze guarantees: for NCM/ridge in
    # Class-IL the matrix is lower-triangular after the freeze so `forgetting`
    # is 0 by construction, but with an adapter the representation keeps moving
    # (L2b) and the router keeps growing (L3), so both can lose old-task
    # accuracy. Report both, computed the same way as S2/S6b.
    forget = [
        max(0.0, float(np.max(R[i : T + 1, i])) - float(R[T, i])) for i in range(T)
    ]
    return {
        "construct": cell["construct"],
        "level": cell["level"],
        "rank": cell["rank"],
        "protos": cell["protos"],
        "top_k": cell["top_k"],
        "seed": cell["seed"],
        "num_tasks": n,
        "classes_per_task": len(tasks[0]["classes"]),
        "accuracy": float(np.mean(R[T, :])),
        "retention": float(np.mean(R[T, :T])) if T > 0 else 0.0,
        "forgetting": float(np.mean(forget)) if forget else 0.0,
        "acc_matrix": R.tolist(),
        "resources": model.resources(),
        "routing": model.routing_stats(tasks, k=3),
    }


def _forgetting(cell: dict) -> float:
    """Recomputed from the stored matrix, so cells written before this metric
    existed still contribute."""
    if "forgetting" in cell:
        return float(cell["forgetting"])
    matrix = cell["acc_matrix"]
    T = len(matrix) - 1
    if T <= 0:
        return 0.0
    forget = [
        max(0.0, max(matrix[t][i] for t in range(i, T + 1)) - matrix[T][i])
        for i in range(T)
    ]
    return float(sum(forget) / len(forget))


def aggregate(cells: list[dict], ranks, prototypes, topks) -> dict:
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

    def pick(construct, **kw):
        return [
            c
            for c in cells
            if c["construct"] == construct
            and all(c[key] == value for key, value in kw.items())
        ]

    def acc(construct, **kw):
        rows = pick(construct, **kw)
        return st([r["accuracy"] for r in rows]) if rows else None

    def first(construct, **kw):
        rows = pick(construct, **kw)
        return rows[0] if rows else None

    out: dict[str, dict] = {}
    for construct in sorted({c["construct"] for c in cells}):
        entry: dict = {"levels": {}, "axes": {}}

        for level in ("L0_ncm", "L1_ridge"):
            rows = pick(construct, level=level)
            entry["levels"][level] = {
                "accuracy": st([r["accuracy"] for r in rows]),
                "resources": rows[0]["resources"] if rows else None,
            }

        parameter = []
        for rank in ranks:
            point = dict(rank=rank, protos=1, top_k=1)
            l2b = acc(construct, level="L2b_shared_seq", **point)
            l3 = acc(construct, level="L3_per_task", **point)
            l4 = acc(construct, level="L4_oracle", **point)
            row = {
                "rank": rank,
                "L2b": l2b,
                "L3": l3,
                "L4": l4,
                "forgetting": {
                    level: st(
                        [_forgetting(r) for r in pick(construct, level=level, **point)]
                    )
                    for level in ("L2b_shared_seq", "L3_per_task", "L4_oracle")
                },
                "resources": (first(construct, level="L3_per_task", **point) or {}).get(
                    "resources"
                ),
            }
            if l3 and l2b:
                row["isolation_realised"] = l3["mean"] - l2b["mean"]
            if l4 and l2b:
                row["isolation_available"] = l4["mean"] - l2b["mean"]
            if l3 and l4:
                row["routing_tax"] = l4["mean"] - l3["mean"]
            if row.get("isolation_available", 0) > 1e-9:
                row["R_iso"] = row["isolation_realised"] / row["isolation_available"]
            # A baseline-free companion to R_iso. `L2b` moves with the rank too
            # (a larger shared adapter is not automatically a better one), so a
            # second ratio against the budget-free NCM anchor separates "the
            # expert bank improved" from "the shared baseline degraded".
            l0 = acc(construct, level="L0_ncm")
            if l3 and l4 and l0 and (l4["mean"] - l0["mean"]) > 1e-9:
                row["R_iso_vs_ncm"] = (l3["mean"] - l0["mean"]) / (
                    l4["mean"] - l0["mean"]
                )
            parameter.append(row)
        entry["axes"]["parameter"] = parameter

        memory = []
        # L4 is oracle-routed, so it does not depend on the router's prototype
        # budget: the tax at a memory point is measured against the same oracle.
        l4_at_operating = acc(
            construct, level="L4_oracle", rank=OPERATING_RANK, protos=1, top_k=1
        )
        for protos in prototypes:
            point = dict(rank=OPERATING_RANK, protos=protos, top_k=1)
            rows = pick(construct, level="L3_per_task", **point)
            l3 = st([r["accuracy"] for r in rows]) if rows else None
            memory.append(
                {
                    "protos": protos,
                    "L3": l3,
                    "forgetting": st([_forgetting(r) for r in rows]),
                    "routing_tax": (
                        l4_at_operating["mean"] - l3["mean"]
                        if l3 and l4_at_operating
                        else None
                    ),
                    "resources": rows[0]["resources"] if rows else None,
                    "recall_at_3": st(
                        [r["routing"].get("task_recall_at_3") for r in rows]
                    ),
                }
            )
        entry["axes"]["memory"] = memory

        active = []
        for top_k in topks:
            point = dict(rank=OPERATING_RANK, protos=1, top_k=top_k)
            rows = pick(construct, level="L3_per_task", **point)
            l3 = st([r["accuracy"] for r in rows]) if rows else None
            active.append(
                {
                    "top_k": top_k,
                    "L3": l3,
                    "forgetting": st([_forgetting(r) for r in rows]),
                    "routing_tax": (
                        l4_at_operating["mean"] - l3["mean"]
                        if l3 and l4_at_operating
                        else None
                    ),
                    "resources": rows[0]["resources"] if rows else None,
                }
            )
        entry["axes"]["active"] = active

        entry["separability"] = st(
            [
                r["separability"]
                for r in pick(
                    construct,
                    level="L3_per_task",
                    rank=OPERATING_RANK,
                    protos=1,
                    top_k=1,
                )
            ]
        )
        entry["n_cells"] = len(pick(construct))
        out[construct] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=SOURCE_CACHE)
    parser.add_argument("--constructs", nargs="*", default=["coherent", "dispersed"])
    parser.add_argument("--ranks", nargs="*", type=int, default=DEFAULT_RANKS)
    parser.add_argument("--prototypes", nargs="*", type=int, default=DEFAULT_PROTOTYPES)
    parser.add_argument("--topks", nargs="*", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--seeds", default="42,1,2")
    parser.add_argument(
        "--sweep-seeds",
        default=None,
        help="seeds for the extra budget points (default: the first --seeds)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument("--max_experts", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/s8")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    sweep_seeds = (
        [int(s) for s in args.sweep_seeds.replace(",", " ").split()]
        if args.sweep_seeds
        else seeds[:1]
    )
    _, source = s2_ladder.load_tasks(args.cache)
    study_path = os.path.join(args.out, "s8_budget_study.json")

    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_func": args.lambda_func,
        "ranks": list(args.ranks),
        "prototypes": list(args.prototypes),
        "topks": list(args.topks),
        "constructs": list(args.constructs),
        "operating_rank": OPERATING_RANK,
        "sweep_seeds": sweep_seeds,
    }
    cells: list[dict] = []
    done: set[tuple] = set()
    if os.path.exists(study_path) and not args.force:
        previous = json.load(open(study_path))
        if previous.get("recipe") == recipe:
            cells = previous.get("cells", [])
            done = {_key(c) for c in cells}
            print(f"[S8] resuming: {len(done)} cells recorded", flush=True)

    def save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "s8_resource_budget",
            "backbone": "vit_b_16+proj768",
            "recipe": recipe,
            "seeds": seeds,
            "cells": cells,
            "aggregate": (
                aggregate(cells, args.ranks, args.prototypes, args.topks)
                if cells
                else {}
            ),
            "contracts": {
                "__".join(str(part) for part in _key(c)): _contract(c) for c in cells
            },
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)

    grid = default_grid(
        args.constructs, args.ranks, args.prototypes, args.topks, seeds, sweep_seeds
    )
    print(
        f"[S8] grid: {len(grid)} cells, {len(done)} already done "
        f"(seeds={seeds}, sweep_seeds={sweep_seeds})",
        flush=True,
    )

    constructions: dict[str, list[dict]] = {}
    for construct in args.constructs:
        constructions[construct] = s6b_difficulty.build_construction(source, construct)
        sep = s6b_difficulty.separability(constructions[construct])
        print(
            f"[S8] {construct}: intra={sep['intra_task_similarity']:.3f} "
            f"cross={sep['cross_task_similarity']:.3f} "
            f"sep={sep['separability']:+.3f}",
            flush=True,
        )

    for cell in grid:
        key = _key(cell)
        if key in done:
            continue
        result = run_cell(cell, constructions[cell["construct"]], args, device)
        result["separability"] = s6b_difficulty.separability(
            constructions[cell["construct"]]
        )["separability"]
        cells.append(result)
        done.add(key)
        save()
        res = result["resources"]
        print(
            f"[S8] {cell['construct']:10s} {cell['level']:16s} r={cell['rank']:<4d} "
            f"p={cell['protos']:<3d} k={cell['top_k']:<2d} seed={cell['seed']:<3d} "
            f"acc={result['accuracy'] * 100:6.2f}%  "
            f"mem={res['memory_bytes'] / 1024:9.1f}KiB "
            f"active={res['active_params']:>9,}",
            flush=True,
        )

    save()
    print(f"[S8] wrote {study_path}", flush=True)
    _print(json.load(open(study_path))["aggregate"])


def _contract(cell: dict) -> dict:
    resources = cell["resources"]
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
                "forgetting": _forgetting(cell),
                "acc_matrix": cell["acc_matrix"],
            },
            "cost": {
                "stored_bytes": resources["memory_bytes"],
                "total_params": resources["total_params"],
            },
        },
        provenance={
            "command": "experiments/s8_budget.py",
            "construct": cell["construct"],
            "rank": cell["rank"],
            "prototypes_per_class": cell["protos"],
            "top_k_experts": cell["top_k"],
            "resources": resources,
            "retention": cell["retention"],
            "forgetting": _forgetting(cell),
            "routing": cell["routing"],
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


def _print(agg: dict) -> None:
    def pct(stats):
        return f"{stats['mean'] * 100:6.2f}" if stats else "   n/a"

    def ratio(value):
        return f"{value:6.3f}" if value is not None else "   n/a"

    print("\n" + "=" * 100)
    print("S8 RESOURCE BUDGET (CIFAR-100, frozen ViT-B/16, Class-IL)")
    print("=" * 100)
    for construct, entry in agg.items():
        sep = entry["separability"]
        print(
            f"\n### {construct}   separability={sep['mean']:+.3f}   "
            f"cells={entry['n_cells']}"
        )
        print(
            f"  references: L0_ncm={pct(entry['levels']['L0_ncm']['accuracy'])}  "
            f"L1_ridge={pct(entry['levels']['L1_ridge']['accuracy'])}"
        )

        print("\n  parameter axis (adapter rank -> expert capacity)")
        print(
            f"  {'rank':>5} {'L2b':>7} {'L3':>7} {'L4':>7} {'tax':>7} "
            f"{'real':>7} {'avail':>7} {'R_iso':>7} {'R_ncm':>7} {'F_L3':>7} "
            f"{'active':>10} {'mem KiB':>9}"
        )
        for row in entry["axes"]["parameter"]:
            res = row["resources"] or {}
            f_l3 = (row.get("forgetting") or {}).get("L3_per_task")
            print(
                f"  {row['rank']:>5} {pct(row['L2b'])} {pct(row['L3'])} "
                f"{pct(row['L4'])} "
                f"{pct({'mean': row['routing_tax']} if row.get('routing_tax') is not None else None)} "
                f"{pct({'mean': row['isolation_realised']} if row.get('isolation_realised') is not None else None)} "
                f"{pct({'mean': row['isolation_available']} if row.get('isolation_available') is not None else None)} "
                f"{ratio(row.get('R_iso'))} "
                f"{ratio(row.get('R_iso_vs_ncm'))} "
                f"{pct(f_l3)} "
                f"{res.get('active_params', 0):>10,} "
                f"{res.get('memory_bytes', 0) / 1024:>9.1f}"
            )

        print("\n  memory axis (prototypes per class -> routing resolution)")
        print(
            f"  {'protos':>7} {'L3':>7} {'tax':>7} {'recall@3':>9} {'F_L3':>7} "
            f"{'mem KiB':>9} {'protos stored':>14}"
        )
        for row in entry["axes"]["memory"]:
            res = row["resources"] or {}
            print(
                f"  {row['protos']:>7} {pct(row['L3'])} "
                f"{pct({'mean': row['routing_tax']} if row.get('routing_tax') is not None else None)} "
                f"{ratio((row['recall_at_3'] or {}).get('mean'))} "
                f"{pct(row.get('forgetting'))} "
                f"{res.get('memory_bytes', 0) / 1024:>9.1f} "
                f"{res.get('num_prototypes', 0):>14,}"
            )

        print("\n  active axis (experts per sample -> inference compute)")
        print(
            f"  {'top_k':>7} {'L3':>7} {'tax':>7} {'F_L3':>7} "
            f"{'active params':>14} {'active experts':>15}"
        )
        for row in entry["axes"]["active"]:
            res = row["resources"] or {}
            print(
                f"  {row['top_k']:>7} {pct(row['L3'])} "
                f"{pct({'mean': row['routing_tax']} if row.get('routing_tax') is not None else None)} "
                f"{pct(row.get('forgetting'))} "
                f"{res.get('active_params', 0):>14,} "
                f"{res.get('active_experts', 0):>15}"
            )


if __name__ == "__main__":
    main()
