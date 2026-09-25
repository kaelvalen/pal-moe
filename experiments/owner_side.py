"""
Owner-side residual: the two-component decomposition of the E2 collapse.

    docs/OWNER_SIDE_PREREG.md

Three independent arms of the pinned contract, re-run fresh:
    all          C1            owner + non-owner evidence gradients
    owner_only   OWNER-ONLY    the owner term only (non-owner paths cut)
    current      C0            no gradient at all (old W frozen)

The header of the study is a conceptual decomposition - never a composition of
arms inside one trajectory:

    Delta_nonowner = OWNER-ONLY - C1        Delta_owner = C0 - OWNER-ONLY

Passive mechanism secondaries, descriptive only: final-model per-task accuracy and
the evidence drift 1 - cos(h_final, h_task-end), measured with torch.no_grad() and
deepcopy snapshots, no autograd.grad anywhere in the study path.

Both anchor sources must match bitwise before anything is read: C1/C0 against the
coupling cells and C1/C0/OWNER-ONLY against the intervention cells.

Usage: python experiments/owner_side.py --device cuda
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
import intervention  # noqa: E402
import s11_confirmatory as s11  # noqa: E402

REGIMES = ["coherent", "dispersed"]
ANCHOR_ARMS = ["all", "current"]
NEW_ARM = "owner_only"
T = 20
COUPLING = "results/coupling/coupling_study.json"
INTERVENTION = "results/intervention/intervention_study.json"
METRIC_KEYS = [
    "accuracy",
    "coverage_at_3",
    "conditional_oracle_at_3",
    "ceiling_at_3",
    "oracle_accuracy",
]


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
    model = e2.E2Model(768, 100, cell_args, device)
    heads, drift = {}, {}

    def on_task_start(m, t):
        return None

    def on_task_end(m, t, st):
        """Passive drift probe (prereg section 3): deepcopy + no_grad only."""
        snap = copy.deepcopy(m)
        with torch.no_grad():
            for j in range(t + 1):
                feats, labels = tasks[j]["splits"]["test"]
                feats, labels = feats.to(m.device), labels.to(m.device)
                per_class = []
                for c in tasks[j]["classes"]:
                    mask = labels == c
                    if not bool(mask.any()):
                        continue
                    z = feats[mask].float()
                    h = e2.normalize(snap.W[j](snap.experts[j].transform(z))).mean(
                        dim=0
                    )
                    if t == j:
                        heads.setdefault(j, {})[c] = h.clone()
                    h_end = heads.get(j, {}).get(c)
                    if h_end is not None:
                        cos = float(
                            F.cosine_similarity(h.unsqueeze(0), h_end.unsqueeze(0))
                        )
                        per_class.append(1.0 - cos)
                if per_class:
                    drift.setdefault(f"task{j}", {})[f"q{t}"] = sum(per_class) / len(
                        per_class
                    )

    started = time.time()
    model.train(
        tasks, seed, hooks={"on_task_start": on_task_start, "on_task_end": on_task_end}
    )
    metrics = model.evaluate(tasks)
    return {
        **metrics,
        "drift": drift,
        "guard": model.guard,
        "seconds": time.time() - started,
    }


def final_drift(cell):
    """Mean over old tasks of the drift at the last task boundary."""
    values = []
    for j in range(T - 1):
        row = cell["drift"].get(f"task{j}", {})
        if f"q{T-1}" in row:
            values.append(row[f"q{T-1}"])
    return sum(values) / len(values) if values else None


def check_anchors(cells):
    """Bitwise: C1/C0 against coupling and intervention, OWNER-ONLY vs intervention."""
    sources = {
        "coupling": {
            (c["arm"], c["regime"], c["seed"]): c
            for c in json.load(open(COUPLING))["cells"]
        },
        "intervention": {
            (c["arm"], c["regime"], c["seed"]): c
            for c in json.load(open(INTERVENTION))["cells"]
        },
    }
    detail, ok = {}, True
    for cell in cells:
        key = (cell["arm"], cell["regime"], cell["seed"])
        wanted = (
            ["coupling", "intervention"]
            if cell["arm"] in ANCHOR_ARMS
            else ["intervention"]
        )
        for source in wanted:
            ref = sources[source][key]
            deltas = {}
            for metric in METRIC_KEYS:
                a, b = cell.get(metric), ref.get(metric)
                deltas[metric] = None if a is None or b is None else abs(a - b)
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
    parser.add_argument("--out", default="results/owner_side")
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "owner_side_study.json")
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]

    audit_rows, audit_pass = intervention.audit(device, args, seed=seeds[0])
    print(f"[OS] cut audit rows={len(audit_rows)} pass={audit_pass}", flush=True)

    study = {
        "schema_version": "1.0",
        "study": "owner_side",
        "prereg": "docs/OWNER_SIDE_PREREG.md",
        "seeds": seeds,
        "cells": [],
        "vetoes": {"cut_audit": {"pass": audit_pass, "rows": audit_rows}},
    }

    def save():
        json.dump(study, open(path, "w"), indent=1)

    if not audit_pass:
        save()
        print("[OS] AUDIT VETO - not executed", flush=True)
        return

    for arm in ANCHOR_ARMS:
        for regime in REGIMES:
            for seed in seeds:
                result = run_cell(arm, regime, seed, args, device)
                study["cells"].append(
                    {"regime": regime, "arm": arm, "seed": seed, **result}
                )
                save()
                print(
                    f"[OS] {arm:10s} {regime:10s} seed={seed:<3d} "
                    f"acc={result['accuracy']*100:6.2f} "
                    f"C@3={result['coverage_at_3']:.4f}",
                    flush=True,
                )

    detail_a, ok_a = check_anchors(study["cells"])
    study["vetoes"]["anchor_invariance_phase_a"] = {"pass": ok_a, "detail": detail_a}
    save()
    if not ok_a:
        print("[OS] ANCHOR VETO (phase A) - not executed", flush=True)
        return
    print("[OS] anchors phase A PASS", flush=True)

    for regime in REGIMES:
        for seed in seeds:
            result = run_cell(NEW_ARM, regime, seed, args, device)
            study["cells"].append(
                {"regime": regime, "arm": NEW_ARM, "seed": seed, **result}
            )
            save()
            print(
                f"[OS] {NEW_ARM:10s} {regime:10s} seed={seed:<3d} "
                f"acc={result['accuracy']*100:6.2f} "
                f"C@3={result['coverage_at_3']:.4f} "
                f"drift_final={final_drift(result):.4f}",
                flush=True,
            )

    detail_b, ok_b = check_anchors(study["cells"])
    study["vetoes"]["anchor_invariance"] = {"pass": ok_b, "detail": detail_b}
    save()
    if not ok_b:
        print("[OS] ANCHOR VETO (phase B) - not executed", flush=True)
        return
    print("[OS] all anchors PASS")
    print("[OS] wrote", path)


if __name__ == "__main__":
    main()
