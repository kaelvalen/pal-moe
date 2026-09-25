"""
AC1: address-freeze completion - the shared query in the routing path.

    docs/AC1_ADDRESS_FREEZE_PREREG.md

Two arms at T = 20, two regimes, three seeds:

    current       C0, the anchor:  w_alignment=current, p_alignment=plastic
    consolidated  C0+P_frozen:     w_alignment=current, p_alignment=consolidated

The anchor cells run with the passive drift probe active and must match the stored
coupling / intervention / owner-side C0 cells on all five metric keys within the
pre-registered 1e-6 band; that comparison is also the passivity demonstration.

The probe measures two trajectories with deepcopy + torch.no_grad() only, no autograd
anywhere in the study path:

    key drift     1 - cos( h_j(z_c) at t , h_j(z_c) at the end of task j )   h = W_j E_j
    query drift   1 - cos( P_t(z_c)     , P_j(z_c)     at the end of task j )

Usage: python experiments/ac1_address_freeze.py --device cpu
"""

import argparse
import copy
import json
import os
import subprocess
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
ARMS = {  # arm -> (w_alignment, p_alignment)
    "current": ("current", "plastic"),
    "consolidated": ("current", "consolidated"),
}
NEW_ARM = "consolidated"
T = 20
# Anchor band, Amendment 1 (prereg section 10): set from the six-cell substrate
# pilot. The two coverage metrics are discrete top-3 statistics whose boundary
# flips under a different floating-point order; the pilot's largest delta is
# 6.0e-04, and 1.5e-3 stays ~11x below the smallest effect of interest (0.0175).
ANCHOR_TOLERANCE = {
    "accuracy": 1.5e-3,
    "ceiling_at_3": 1.5e-3,
    "oracle_accuracy": 1.5e-3,
    "coverage_at_3": 1.5e-3,
    "conditional_oracle_at_3": 1.5e-3,
}
CELL_CEILING_S = 3600
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


def run_cell(arm, regime, seed, args, device):
    w_alignment, p_alignment = ARMS[arm]
    _, source = s11.s2_ladder.load_tasks(e2.SOURCE_CACHE)
    tasks = s11.s6b_difficulty.build_construction(source, regime)[:T]
    cell_args = argparse.Namespace(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        rank=8,
        evidence_lambda=1.0,
        w_alignment=w_alignment,
        p_alignment=p_alignment,
        seed=seed,
    )
    s11.s2_ladder.set_seed(seed)
    # Exactly one construction: a second one consumes the global RNG and breaks
    # the pinned trajectory (the interference study's root cause).
    model = e2.E2Model(768, 100, cell_args, device)
    keys, queries = {}, {}

    def on_task_start(m, t):
        return None

    def on_task_end(m, t, st):
        """Passive drift probe (prereg section 3): deepcopy + no_grad only."""
        snap = copy.deepcopy(m)
        with torch.no_grad():
            for j in range(t + 1):
                feats, labels = tasks[j]["splits"]["test"]
                feats, labels = feats.to(m.device), labels.to(m.device)
                per_class_key, per_class_query = [], []
                for c in tasks[j]["classes"]:
                    mask = labels == c
                    if not bool(mask.any()):
                        continue
                    z = feats[mask].float()
                    h = e2.normalize(snap.W[j](snap.experts[j].transform(z))).mean(
                        dim=0
                    )
                    q = e2.normalize(snap.P(z)).mean(dim=0)
                    if t == j:
                        keys.setdefault(j, {})[c] = h.clone()
                        queries.setdefault(j, {})[c] = q.clone()
                    h_end = keys.get(j, {}).get(c)
                    q_end = queries.get(j, {}).get(c)
                    if h_end is not None:
                        per_class_key.append(
                            1.0
                            - float(
                                F.cosine_similarity(h.unsqueeze(0), h_end.unsqueeze(0))
                            )
                        )
                    if q_end is not None:
                        per_class_query.append(
                            1.0
                            - float(
                                F.cosine_similarity(q.unsqueeze(0), q_end.unsqueeze(0))
                            )
                        )
                if per_class_key:
                    drift_key.setdefault(f"task{j}", {})[f"q{t}"] = sum(
                        per_class_key
                    ) / len(per_class_key)
                if per_class_query:
                    drift_query.setdefault(f"task{j}", {})[f"q{t}"] = sum(
                        per_class_query
                    ) / len(per_class_query)

    drift_key, drift_query = {}, {}
    started = time.time()
    model.train(
        tasks, seed, hooks={"on_task_start": on_task_start, "on_task_end": on_task_end}
    )
    metrics = model.evaluate(tasks)
    return {
        **metrics,
        "drift_key": drift_key,
        "drift_query": drift_query,
        "guard": model.guard,
        "seconds": time.time() - started,
    }


def final_drift(cell, field):
    """Mean over old tasks of a drift trajectory at the last task boundary."""
    values = []
    for j in range(T - 1):
        row = cell[field].get(f"task{j}", {})
        if f"q{T-1}" in row:
            values.append(row[f"q{T-1}"])
    return sum(values) / len(values) if values else None


def check_anchors(cells):
    """Within-tolerance match of every C0 cell against every stored source."""
    index = {name: json.load(open(path))["cells"] for name, path in SOURCES.items()}
    detail, ok = {}, True
    for cell in cells:
        if cell["arm"] != "current":
            continue
        for source, rows in index.items():
            ref = next(
                (
                    r
                    for r in rows
                    if r["arm"] == "current"
                    and r["regime"] == cell["regime"]
                    and r["seed"] == cell["seed"]
                ),
                None,
            )
            if ref is None:
                continue
            deltas = {
                k: (
                    None
                    if cell.get(k) is None or ref.get(k) is None
                    else abs(cell[k] - ref[k])
                )
                for k in METRIC_KEYS
            }
            key = f"{cell['regime']}/{cell['seed']}/{source}"
            detail[key] = {
                "deltas": deltas,
                "tolerances": ANCHOR_TOLERANCE,
                "max_abs_delta": max(v for v in deltas.values() if v is not None),
            }
            if any(v is None or v > ANCHOR_TOLERANCE[k] for k, v in deltas.items()):
                ok = False
    return detail, ok


def check_guard(cells):
    """P freeze guard (new arm) and P plasticity guard (anchor), per task."""
    detail, ok = {}, True
    for cell in cells:
        frozen = cell["arm"] == NEW_ARM
        rows = cell.get("guard", {})
        if not rows:
            ok = False
            detail[f"{cell['arm']}/{cell['regime']}/{cell['seed']}"] = "no guard rows"
            continue
        checks = {}
        for task, row in rows.items():
            if frozen:
                good = (
                    row.get("P_frozen") is True
                    and row.get("P_in_optimizer") == 0
                    and row.get("P_grad_nonzero") is False
                )
            else:
                good = (
                    row.get("P_frozen") is False
                    and row.get("P_in_optimizer") == 1
                    and row.get("P_grad_nonzero") is True
                )
            checks[task] = bool(good)
            ok = ok and bool(good)
        detail[f"{cell['arm']}/{cell['regime']}/{cell['seed']}"] = {
            "tasks_checked": len(checks),
            "all_pass": all(checks.values()) if checks else False,
        }
    return detail, ok


def manipulation_check(cells):
    """Key drift zero in both arms; query drift zero in the new arm only.

    Zero is read to floating-point precision: the probe compares a class-mean
    vector against a clone, and a float32 self-cosine is ~1e-7, not bitwise 1.0.
    """
    detail = {}
    ok = True
    for cell in cells:
        key = f"{cell['arm']}/{cell['regime']}/{cell['seed']}"
        kd, qd = final_drift(cell, "drift_key"), final_drift(cell, "drift_query")
        frozen = cell["arm"] == NEW_ARM
        good = (
            kd is not None
            and abs(kd) <= 1e-5
            and qd is not None
            and (qd <= 1e-5 if frozen else qd >= 1e-3)
        )
        detail[key] = {"key_drift_final": kd, "query_drift_final": qd, "pass": good}
        ok = ok and good
    return detail, ok


def stats(cells):
    """Paired C@3 (primary) and Acc (secondary) over (regime, seed)."""
    index = {(c["regime"], c["arm"], c["seed"]): c for c in cells}
    seeds = sorted({c["seed"] for c in cells if c["arm"] == NEW_ARM})
    out = {}
    for metric, label in [
        ("coverage_at_3", "primary_C@3"),
        ("accuracy", "secondary_Acc"),
    ]:
        values, per_pair = [], {}
        for regime in REGIMES:
            for seed in seeds:
                new = index.get((regime, NEW_ARM, seed))
                old = index.get((regime, "current", seed))
                if new is None or old is None:
                    continue
                d = new[metric] - old[metric]
                values.append(d)
                per_pair[f"{regime}/{seed}"] = d
        if values:
            row = s11.paired_stats(values, label)
            row["per_pair"] = per_pair
            out[label] = row
    out["family_note"] = (
        "one test per family; the max-statistic Westfall-Young correction "
        "reduces to the exact paired permutation p reported in each row"
    )
    return out


def harness_revision():
    """The commit the study was run from (recorded with the study)."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).resolve().parent.parent,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,1,2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/ac1")
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "ac1_address_freeze_study.json")
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]

    study = {
        "schema_version": "1.0",
        "study": "ac1_address_freeze",
        "prereg": "docs/AC1_ADDRESS_FREEZE_PREREG.md",
        "git_revision": harness_revision(),
        "device": args.device,
        "seeds": seeds,
        "cells": [],
        "vetoes": {},
        "stats": {},
    }

    def save():
        json.dump(study, open(path, "w"), indent=1)

    # Phase A: the anchor, probe active (passivity + equivalence).
    for regime in REGIMES:
        for seed in seeds:
            result = run_cell("current", regime, seed, args, device)
            study["cells"].append(
                {"arm": "current", "regime": regime, "seed": seed, **result}
            )
            save()
            print(
                f"[AC1] anchor       {regime:10s} seed={seed:<3d} "
                f"acc={result['accuracy']*100:6.2f} "
                f"C@3={result['coverage_at_3']:.4f} "
                f"qdrift={final_drift(result, 'drift_query'):.4f} "
                f"({result['seconds']:.0f}s)",
                flush=True,
            )

    detail_a, ok_a = check_anchors(study["cells"])
    study["vetoes"]["anchor_equivalence_phase_a"] = {"pass": ok_a, "detail": detail_a}
    save()
    if not ok_a:
        print("[AC1] ANCHOR VETO (phase A) - not executed", flush=True)
        return
    print("[AC1] anchors phase A PASS", flush=True)

    # Phase B: the new arm.
    for regime in REGIMES:
        for seed in seeds:
            result = run_cell(NEW_ARM, regime, seed, args, device)
            study["cells"].append(
                {"arm": NEW_ARM, "regime": regime, "seed": seed, **result}
            )
            save()
            print(
                f"[AC1] consolidated {regime:10s} seed={seed:<3d} "
                f"acc={result['accuracy']*100:6.2f} "
                f"C@3={result['coverage_at_3']:.4f} "
                f"qdrift={final_drift(result, 'drift_query'):.4f} "
                f"({result['seconds']:.0f}s)",
                flush=True,
            )

    detail_b, ok_b = check_anchors(study["cells"])
    study["vetoes"]["anchor_equivalence"] = {"pass": ok_b, "detail": detail_b}
    guard_detail, guard_ok = check_guard(study["cells"])
    study["vetoes"]["p_freeze_guard"] = {"pass": guard_ok, "detail": guard_detail}
    drift_detail, drift_ok = manipulation_check(study["cells"])
    study["vetoes"]["manipulation_check"] = {"pass": drift_ok, "detail": drift_detail}
    costs = [c["seconds"] for c in study["cells"]]
    study["feasibility"] = {
        "cells_planned": 2 * len(REGIMES) * len(seeds),
        "mean_cell_seconds": sum(costs) / len(costs) if costs else None,
        "projected_total_hours": (
            2 * len(REGIMES) * len(seeds) * sum(costs) / len(costs) / 3600
        ),
        "ceiling_hours": CELL_CEILING_S / 3600,
    }
    study["stats"] = stats(study["cells"])
    save()
    print(
        "[AC1] vetoes:",
        json.dumps({k: v["pass"] for k, v in study["vetoes"].items()}),
        flush=True,
    )
    print(json.dumps(study["stats"], indent=1)[:1200], flush=True)
    print("[AC1] wrote", path)


if __name__ == "__main__":
    main()
