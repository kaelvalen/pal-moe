"""
Coupling ablation: accumulated evidence re-alignment. docs/COUPLING_PREREG.md

    C0  current W_t trainable, W_1..W_{t-1} frozen     (--w_alignment current)
    C1  all W_1..W_t trainable                          (--w_alignment all, the E2 contract)

Everything else is the pinned E2 contract. Feasibility is a declared constraint:
12 cells against a 12 h ceiling, and the smoke's measured cost decided execution.

Usage: python experiments/coupling_ablation.py --seeds 42,1,2 --device cuda
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import e2_evidence as e2  # noqa: E402
import s11_confirmatory as s11  # noqa: E402

REGIMES = ["coherent", "dispersed"]
ARMS = ["all", "current"]
E2_ANCHOR = {  # E2 lambda = 1, seed 42, coherent; the C1 arm must reproduce it
    "accuracy": 0.1304,
    "coverage_at_3": 0.9041,
}
CELL_CEILING_S = 12 * 3600


def run(regime, arm, seed, args, device):
    _, source = s11.s2_ladder.load_tasks(e2.SOURCE_CACHE)
    tasks = s11.s6b_difficulty.build_construction(source, regime)
    cell_args = argparse.Namespace(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        rank=8,
        evidence_lambda=1.0,
        w_alignment=arm,
        seed=seed,
    )
    s11.s2_ladder.set_seed(seed)
    model = e2.E2Model(768, 100, cell_args, device)
    started = time.time()
    model.train(tasks, seed)
    metrics = model.evaluate(tasks)
    return {**metrics, "guard": model.guard, "seconds": time.time() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,1,2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/coupling")
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    path = os.path.join(args.out, "coupling_study.json")
    cells, done = [], set()
    if os.path.exists(path):
        previous = json.load(open(path))
        cells = previous.get("cells", [])
        done = {(c["regime"], c["arm"], c["seed"]) for c in cells}

    def save():
        costs = [c["seconds"] for c in cells]
        payload = {
            "schema_version": "1.0",
            "study": "coupling_ablation",
            "prereg": "docs/COUPLING_PREREG.md",
            "seeds": seeds,
            "cells": cells,
            "feasibility": {
                "cells_planned": len(REGIMES) * len(ARMS) * len(seeds),
                "mean_cell_seconds": sum(costs) / len(costs) if costs else None,
                "projected_total_hours": (
                    len(REGIMES)
                    * len(ARMS)
                    * len(seeds)
                    * sum(costs)
                    / len(costs)
                    / 3600
                    if costs
                    else None
                ),
                "ceiling_hours": CELL_CEILING_S / 3600,
            },
            "vetoes": vetoes(cells, seeds),
        }
        json.dump(payload, open(path, "w"), indent=1)

    for regime in REGIMES:
        for arm in ARMS:
            for seed in seeds:
                if (regime, arm, seed) in done:
                    continue
                result = run(regime, arm, seed, args, device)
                cells.append({"regime": regime, "arm": arm, "seed": seed, **result})
                save()
                print(
                    f"[CPL] {regime:10s} C{1 if arm == 'all' else 0} seed={seed:<3d} "
                    f"acc={result['accuracy']*100:6.2f} "
                    f"C@3={result['coverage_at_3']:.4f} "
                    f"oracle@3={result['conditional_oracle_at_3'] or float('nan'):.4f} "
                    f"({result['seconds']:.0f}s)",
                    flush=True,
                )
    save()
    print("[CPL] vetoes:", json.dumps(vetoes(cells, seeds), indent=1)[:600])


def vetoes(cells, seeds):
    index = {(c["regime"], c["arm"], c["seed"]): c for c in cells}
    anchor = {}
    for seed in seeds:
        c1 = index.get(("coherent", "all", seed))
        if c1:
            anchor[seed] = (
                {
                    "d_accuracy": abs(E2_ANCHOR["accuracy"] - c1["accuracy"]),
                    "d_coverage": abs(E2_ANCHOR["coverage_at_3"] - c1["coverage_at_3"]),
                }
                if seed == 42
                else "n/a (anchor is seed 42)"
            )
    c0 = [c for c in cells if c["arm"] == "current" and c.get("guard")]
    isolation = (
        all(
            row.get("current_W_grad_nonzero")
            and row.get("old_W_frozen")
            and row.get("old_W_with_gradient") == 0
            and row.get("previous_experts_in_optimizer") == 0
            for c in c0
            for row in c["guard"].values()
        )
        if c0
        else None
    )
    costs = [c["seconds"] for c in cells]
    return {
        "C1_anchor_seed42": anchor.get(42),
        "C0_gradient_isolation": isolation,
        "feasibility_ok": (
            bool(
                costs
                and max(costs) * len(REGIMES) * len(ARMS) * len(seeds) / len(costs)
                < CELL_CEILING_S
            )
            if costs
            else None
        ),
    }


if __name__ == "__main__":
    main()
