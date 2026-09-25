"""
Interference mechanism: new prototypes against old projections.

    docs/INTERFERENCE_PREREG.md

Passive instrumentation on the pinned C1 (all-W) trajectory: no manipulation of
the training, only a loss-based probe that runs in torch.no_grad() on a deepcopy
(no autograd, no .grad, no optimizer access - Amendment 1), plus W snapshots and
an expert-count ladder.

Measurements per task q and old projection W_j (j < q): ||delta_W_j||, and the
owner/non-owner loss split of L_q by prototype ownership.

Usage: python experiments/interference.py --device cuda
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import e2_evidence as e2  # noqa: E402
import s11_confirmatory as s11  # noqa: E402

TS = [2, 4, 8, 12, 20]
REGIMES = ["coherent", "dispersed"]


def probe(model, task_index):
    """Loss-based owner/non-owner asymmetry (Amendment 1).

    No autograd, no retain_graph, no .grad access: the whole probe runs in
    `torch.no_grad()` on a deepcopy, so it cannot touch the training trajectory.
    """
    if task_index == 0 or not model.prototypes:
        return {}
    terms = []
    with torch.no_grad():
        for z_p, owner in model.prototypes:
            q = e2.normalize(model.P(z_p))
            scores = []
            for j in range(len(model.experts)):
                h = e2.normalize(
                    model.W[j](model.experts[j].transform(z_p.unsqueeze(0))[0])
                )
                scores.append(torch.dot(q, h))
            loss_p = F.cross_entropy(
                torch.stack(scores).unsqueeze(0),
                torch.tensor([owner], device=model.device),
            )
            terms.append((owner, float(loss_p)))
    out = {}
    for j in range(task_index):
        owner_mass = sum(loss for own, loss in terms if own == j)
        nonowner_mass = sum(loss for own, loss in terms if own != j)
        out[f"W{j}"] = {
            "owner": owner_mass,
            "nonowner": nonowner_mass,
            "ratio": nonowner_mass / max(owner_mass, 1e-12),
        }
    return out


def run(
    regime,
    T,
    seed,
    args,
    device,
    probe_enabled=True,
    norm_enabled=True,
    hooks_enabled=True,
):
    """Run one cell of the ladder.

    The three `*_enabled` flags exist only for the implementation-invariance
    bisect (probe body / ||delta_W_j|| / the hook mechanism itself). The study's
    ladder always runs with all three at their defaults.
    """
    _, source = s11.s2_ladder.load_tasks(e2.SOURCE_CACHE)
    tasks = s11.s6b_difficulty.build_construction(source, regime)[:T]
    cell_args = argparse.Namespace(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        rank=8,
        evidence_lambda=1.0,
        w_alignment="all",
        seed=seed,
    )
    s11.s2_ladder.set_seed(seed)
    # Exactly one construction. A second one consumes the global RNG and starts
    # the pinned trajectory from different weights - that was the anchor bug.
    model = e2.E2Model(768, 100, cell_args, device)
    probes, rewrite = {}, {}

    def on_task_start(m, t):
        return {
            "probes": probe(copy.deepcopy(m), t) if (t > 0 and probe_enabled) else None,
            "snaps": {j: m.W[j].weight.detach().clone() for j in range(t + 1)},
        }

    def on_task_end(m, t, st):
        if st["probes"]:
            probes[f"q{t}"] = st["probes"]
        if norm_enabled:
            for j in range(t + 1):
                rewrite.setdefault(f"after_t{t}", {})[f"W{j}"] = float(
                    (m.W[j].weight.detach() - st["snaps"][j]).norm()
                )

    hooks = (
        {"on_task_start": on_task_start, "on_task_end": on_task_end}
        if hooks_enabled
        else None
    )
    model.train(tasks, seed, hooks=hooks)
    metrics = model.evaluate(tasks)
    return {**metrics, "probes": probes, "rewrite": rewrite, "T": T}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,1,2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/interference")
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    path = os.path.join(args.out, "interference_study.json")
    cells = []
    for regime in REGIMES:
        for T in TS:
            for seed in seeds:
                result = run(regime, T, seed, args, device)
                cells.append({"regime": regime, "seed": seed, **result})
                json.dump(
                    {
                        "schema_version": "1.0",
                        "study": "interference",
                        "prereg": "docs/INTERFERENCE_PREREG.md",
                        "seeds": seeds,
                        "cells": cells,
                    },
                    open(path, "w"),
                    indent=1,
                )
                ratios = [
                    v["ratio"] for v in result["probes"].get(f"q{T-1}", {}).values()
                ]
                print(
                    f"[INT] {regime:10s} T={T:<3d} seed={seed:<3d} "
                    f"acc={result['accuracy']*100:6.2f} C@3={result['coverage_at_3']:.4f} "
                    f"nonowner/owner={sum(ratios)/max(len(ratios),1):.3f}",
                    flush=True,
                )
    print("[INT] wrote", path)


if __name__ == "__main__":
    main()
