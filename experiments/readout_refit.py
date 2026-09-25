"""
Readout refit / replay: decode mismatch or representation loss?

    docs/REFIT_PREREG.md

Training is pinned and untouched: the three arms re-run fresh and their R0
evaluations must equal the stored cells bitwise. After training, each trained
model is evaluated under two post-hoc readout treatments on a deepcopy:

    R1  refit-current  the readout refitted on the last task's training evidence
    R2  refit-replay   the readout refitted on every task's training evidence

The refit protocol is identical for R1 and R2 (data definition, budget,
initialisation, stopping, batch order); it never sees a test split, and
`coverage_at_3` must stay bitwise identical across R0/R1/R2 - the check that only
`g` changed.

Usage: python experiments/readout_refit.py --device cuda
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import e2_evidence as e2  # noqa: E402
import s11_confirmatory as s11  # noqa: E402

REGIMES = ["coherent", "dispersed"]
ANCHOR_ARMS = ["all", "current"]
NEW_ARM = "owner_only"
T = 20
NUM_CLASSES = 100
REFIT_EPOCHS = 10
REFIT_LR = 1e-3
SOURCES = {
    "coupling": "results/coupling/coupling_study.json",
    "intervention": "results/intervention/intervention_study.json",
    "owner_side": "results/owner_side/owner_side_study.json",
}
METRIC_KEYS = [
    "accuracy",
    "coverage_at_3",
    "conditional_oracle_at_3",
    "ceiling_at_3",
    "oracle_accuracy",
]


def refit_readout(model, tasks, task_indices, seed, batch_size):
    """The pinned post-hoc refit (prereg section 2.1); identical for R1 and R2.

    Only the data scope changes. The refit never sees a test split and runs on a
    deepcopy of the trained readout.
    """
    readout = copy.deepcopy(model.readout)
    hs, ys = [], []
    with torch.no_grad():
        for j in task_indices:
            feats, labels = tasks[j]["splits"]["train"]
            feats, labels = feats.to(model.device), labels.to(model.device)
            h = e2.normalize(model.W[j](model.experts[j].transform(feats)))
            hs.append(h)
            ys.append(labels)
    x, y = torch.cat(hs), torch.cat(ys)
    handle = s11.s2_ladder.trainable_hook(readout, list(range(NUM_CLASSES)))
    optimizer = torch.optim.Adam(readout.parameters(), lr=REFIT_LR)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(REFIT_EPOCHS):
        for z, yy in s11.s2_ladder.iter_batches(x, y, batch_size, generator):
            loss = F.cross_entropy(readout.predict(z), yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    handle.remove()
    return readout


def run_cell(arm, regime, seed, args, device):
    _, source = s11.s2_ladder.load_tasks(e2.SOURCE_CACHE)
    tasks = s11.s6b_difficulty.build_construction(source, regime)[:T]
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
    # Exactly one construction: a second one consumes the global RNG and breaks
    # the pinned trajectory (the interference study's root cause).
    model = e2.E2Model(768, NUM_CLASSES, cell_args, device)
    started = time.time()
    model.train(tasks, seed)  # no instrumentation: the pinned training path
    r0 = model.evaluate(tasks)

    r1_readout = refit_readout(model, tasks, [T - 1], seed, args.batch_size)
    r2_readout = refit_readout(model, tasks, list(range(T)), seed, args.batch_size)
    snap = copy.deepcopy(model)
    snap.readout = r1_readout
    r1 = snap.evaluate(tasks)
    snap.readout = r2_readout
    r2 = snap.evaluate(tasks)

    invariance = r0["coverage_at_3"] == r1["coverage_at_3"] == r2["coverage_at_3"]
    return {
        "R0": r0,
        "R1": r1,
        "R2": r2,
        "C3_invariance": bool(invariance),
        "guard": model.guard,
        "seconds": time.time() - started,
    }


def check_anchors(cells):
    """Bitwise R0: five metrics against every source; per-task accuracy vs owner_side."""
    sources = {
        name: {
            (c["arm"], c["regime"], c["seed"]): c
            for c in json.load(open(path))["cells"]
        }
        for name, path in SOURCES.items()
    }
    detail, ok = {}, True
    for cell in cells:
        key = (cell["arm"], cell["regime"], cell["seed"])
        wanted = (
            ["coupling", "intervention", "owner_side"]
            if cell["arm"] in ANCHOR_ARMS
            else ["intervention", "owner_side"]
        )
        for source in wanted:
            ref = sources[source][key]
            deltas = {}
            for metric in METRIC_KEYS:
                a, b = cell["R0"].get(metric), ref.get(metric)
                deltas[metric] = None if a is None or b is None else abs(a - b)
            if source == "owner_side":
                a, b = cell["R0"].get("per_task_accuracy"), ref.get("per_task_accuracy")
                deltas["per_task_accuracy"] = (
                    None
                    if a is None or b is None
                    else max(abs(u - v) for u, v in zip(a, b))
                )
            detail[f"{cell['arm']}/{cell['regime']}/{cell['seed']}/{source}"] = deltas
            if any(v is None or v != 0.0 for v in deltas.values()):
                ok = False
    return detail, ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,1,2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/refit")
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "refit_study.json")
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]

    study = {
        "schema_version": "1.0",
        "study": "refit",
        "prereg": "docs/REFIT_PREREG.md",
        "seeds": seeds,
        "recipe": {
            "refit_epochs": REFIT_EPOCHS,
            "refit_lr": REFIT_LR,
            "refit_batch_size": args.batch_size,
            "refit_data": "training features through the owner expert",
            "R1_scope": "last task only",
            "R2_scope": "all tasks",
        },
        "cells": [],
        "vetoes": {},
    }

    def save():
        json.dump(study, open(path, "w"), indent=1)

    for arm in ANCHOR_ARMS:
        for regime in REGIMES:
            for seed in seeds:
                result = run_cell(arm, regime, seed, args, device)
                study["cells"].append(
                    {"regime": regime, "arm": arm, "seed": seed, **result}
                )
                save()
                print(
                    f"[RF] {arm:10s} {regime:10s} seed={seed:<3d} "
                    f"R0 acc={result['R0']['accuracy']*100:6.2f} "
                    f"C@3={result['R0']['coverage_at_3']:.4f} | "
                    f"R1 acc={result['R1']['accuracy']*100:6.2f} | "
                    f"R2 acc={result['R2']['accuracy']*100:6.2f} | "
                    f"C3inv={result['C3_invariance']}",
                    flush=True,
                )

    detail_a, ok_a = check_anchors(study["cells"])
    study["vetoes"]["anchor_invariance_phase_a"] = {"pass": ok_a, "detail": detail_a}
    save()
    if not ok_a:
        print("[RF] ANCHOR VETO (phase A) - not executed", flush=True)
        return
    print("[RF] anchors phase A PASS", flush=True)

    for regime in REGIMES:
        for seed in seeds:
            result = run_cell(NEW_ARM, regime, seed, args, device)
            study["cells"].append(
                {"regime": regime, "arm": NEW_ARM, "seed": seed, **result}
            )
            save()
            print(
                f"[RF] {NEW_ARM:10s} {regime:10s} seed={seed:<3d} "
                f"R0 acc={result['R0']['accuracy']*100:6.2f} "
                f"C@3={result['R0']['coverage_at_3']:.4f} | "
                f"R1 acc={result['R1']['accuracy']*100:6.2f} | "
                f"R2 acc={result['R2']['accuracy']*100:6.2f} | "
                f"C3inv={result['C3_invariance']}",
                flush=True,
            )

    detail_b, ok_b = check_anchors(study["cells"])
    study["vetoes"]["anchor_invariance"] = {"pass": ok_b, "detail": detail_b}
    c3_ok = all(c["C3_invariance"] for c in study["cells"])
    study["vetoes"]["c3_invariance"] = {"pass": c3_ok}
    save()
    if not ok_b:
        print("[RF] ANCHOR VETO (phase B) - not executed", flush=True)
        return
    if not c3_ok:
        print("[RF] C@3 INVARIANCE VETO - not executed", flush=True)
        return
    print("[RF] all anchors PASS | C@3 invariance PASS")
    print("[RF] wrote", path)


if __name__ == "__main__":
    main()
