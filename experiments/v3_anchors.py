"""
v3 restructure: re-run the stored anchors and report every delta.

A refactor that changes a number is a bug, not a finding. This runner re-executes
three stored studies cell by cell through the *current* code and compares against
the stored JSONs (which stay untracked in results/):

    s11     S11 E0 = `L3_per_task`, T=20, rank 8, 2 constructions x 6 seeds.
            Band 0: `accuracy` and the full `acc_matrix` must be bitwise.
    ac3     AC3 cells, 2 regimes x 3 seeds: `fixed_proto` and `bilinear` scalar
            metrics and per-task accuracies. Band 1e-6.
    etid2   E-TID2 cells, 2 regimes x 6 seeds: all five arms, both coverages and
            the three guards. Band 1e-6 (the `--api` variant runs the ridge arms
            through the v3 float64 medium path and only *reports* those deltas).

Usage:
    LD_LIBRARY_PATH=/run/opengl-driver/lib \
        .venv/bin/python experiments/v3_anchors.py --only s11,ac3,etid2 --device cuda
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

S11_JSON = "results/s11/s11_confirmatory_study.json"
AC3_JSON = "results/ac3/ac3_address_space_study.json"
ETID2_JSON = "results/e_tid2/e_tid2_ridge_router.json"
BANDS = {"s11": 0.0, "ac3": 1e-6, "etid2": 1e-6}
ETID2_ARMS = ("proto", "ridge_routed", "ridge_masked", "ridge_alone", "oracle")
ETID2_COV = ("proto_C@1", "proto_C@3", "ridge_C@1", "ridge_C@3")
API_BANK_FIELDS = (
    "delta_proto",
    "delta_oracle",
    "delta_proto_C@1",
    "delta_proto_C@3",
    "delta_guards_G1_G2",
)
API_RIDGE_FIELDS = (
    "delta_ridge_routed",
    "delta_ridge_masked",
    "delta_ridge_alone",
    "delta_ridge_C@1",
    "delta_ridge_C@3",
)


def _max_abs(a, b) -> float:
    """Max |a - b| over matching numeric leaves; lists compared elementwise."""
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if a == b else math.inf
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return math.inf
        return max((_max_abs(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) & set(b)
        return max((_max_abs(a[k], b[k]) for k in keys), default=0.0)
    return 0.0 if a == b else math.inf


def check_s11(seeds, args, device) -> list[dict]:
    import s11_confirmatory as s11

    stored = json.load(open(S11_JSON))
    index = {
        (c["construct"], c["seed"]): c
        for c in stored["cells"]
        if c["part"] == "B"
        and c["level"] == "L3_per_task"
        and c["rank"] == 8
        and c["protos"] == 1
        and c["num_tasks"] == 20
    }
    run_args = argparse.Namespace(
        epochs=10, lr=1e-3, batch_size=128, lambda_func=1.0, seed=None
    )
    rows, source_cache = [], {}
    for construct in ("coherent", "dispersed"):
        for seed in seeds:
            ref = index[(construct, seed)]
            cell = {
                k: ref[k]
                for k in (
                    "part",
                    "dataset",
                    "construct",
                    "level",
                    "rank",
                    "protos",
                    "top_k",
                    "num_tasks",
                    "seed",
                )
            }
            got = s11.run_cell(cell, run_args, device, source_cache)
            rows.append(
                {
                    "construct": construct,
                    "seed": seed,
                    "stored": ref["accuracy"],
                    "measured": got["accuracy"],
                    "delta_accuracy": got["accuracy"] - ref["accuracy"],
                    "delta_acc_matrix": _max_abs(got["acc_matrix"], ref["acc_matrix"]),
                }
            )
            print(
                f"[s11] {construct:9s} {seed:<3d} stored {ref['accuracy']:.10f} "
                f"measured {got['accuracy']:.10f} "
                f"d={rows[-1]['delta_accuracy']:+.2e} dR={rows[-1]['delta_acc_matrix']:.2e}",
                flush=True,
            )
    return rows


def check_ac3(seeds, args, device) -> list[dict]:
    import ac3_address_space as ac3

    stored = json.load(open(AC3_JSON))
    index = {(c["regime"], c["seed"]): c for c in stored["cells"]}
    run_args = argparse.Namespace(epochs=10, lr=1e-3, batch_size=128)
    rows = []
    for regime in ("coherent", "dispersed"):
        for seed in seeds:
            ref = index[(regime, seed)]
            got = ac3.run_cell(regime, seed, run_args, device)
            row = {"regime": regime, "seed": seed}
            for arm in ("fixed_proto", "bilinear", "anchor_metrics"):
                row[f"delta_{arm}"] = _max_abs(got[arm], ref[arm])
            row["router_param_count"] = got["router_param_count"]
            row["fixed_accuracy"] = got["fixed_proto"]["accuracy"]
            rows.append(row)
            print(
                f"[ac3] {regime:9s} {seed:<3d} fixed acc {row['fixed_accuracy']:.4f} "
                f"d_fixed={row['delta_fixed_proto']:.2e} d_bil={row['delta_bilinear']:.2e}",
                flush=True,
            )
    return rows


def check_etid2(seeds, args, device, api: bool = False) -> list[dict]:
    import e_tid2_ridge_router as e2r
    import s2_ladder

    stored = json.load(open(ETID2_JSON))
    index = {(c["regime"], c["seed"]): c for c in stored["cells"]}
    run_args = argparse.Namespace(
        epochs=10, lr=1e-3, batch_size=128, lambda_func=1.0, ridge=1.0, seed=None
    )
    _, base = s2_ladder.load_tasks(e2r.CACHE)
    rows = []
    for regime in e2r.REGIMES:
        for seed in seeds:
            run_args.seed = seed
            ref = index[(regime, seed)]
            if api:
                import v3_etid2_api

                got = v3_etid2_api.run_cell(regime, seed, run_args, device, base)
            else:
                got = e2r.run_cell(regime, seed, run_args, device, base)
            row = {"regime": regime, "seed": seed}
            for k in ETID2_ARMS + ETID2_COV:
                row[k] = got[k]
                row[f"delta_{k}"] = got[k] - ref[k]
            row["guards"] = got["guards"]
            if "v3_guards" in got:
                row["v3_guards"] = got["v3_guards"]
            row["delta_guards_G1_G2"] = max(
                abs(got["guards"]["G1_max_abs"] - ref["guards"]["G1_max_abs"]),
                abs(got["guards"]["G2_max_abs"] - ref["guards"]["G2_max_abs"]),
            )
            rows.append(row)
            worst = max(abs(row[f"delta_{k}"]) for k in ETID2_ARMS + ETID2_COV)
            print(
                f"[etid2{'-api' if api else ''}] {regime:9s} {seed:<3d} "
                f"max|d| {worst:.2e} G3 mismatch {got['guards']['G3_argmax_mismatch']}",
                flush=True,
            )
    return rows


def verdict(name, rows, band) -> dict:
    fields = [k for k in rows[0] if k.startswith("delta_")] if rows else []
    worst = max((abs(r[k]) for r in rows for k in fields), default=0.0)
    return {
        "cells": len(rows),
        "band": band,
        "max_abs_delta": worst,
        "pass": worst <= band,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default="s11,ac3,etid2")
    p.add_argument("--seeds6", default="42,1,2,3,4,5")
    p.add_argument("--seeds3", default="42,1,2")
    p.add_argument(
        "--api",
        action="store_true",
        help="run E-TID2 through the v3 API (float64 medium path)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/v3/anchors.json")
    args = p.parse_args()
    device = torch.device(args.device)
    s6 = [int(s) for s in args.seeds6.split(",")]
    s3 = [int(s) for s in args.seeds3.split(",")]

    report, started = {}, time.time()
    for name in args.only.split(","):
        if name == "s11":
            rows = check_s11(s6, args, device)
        elif name == "ac3":
            rows = check_ac3(s3, args, device)
        elif name == "etid2":
            rows = check_etid2(s6, args, device, api=args.api)
        else:
            raise SystemExit(f"unknown anchor {name}")
        key = f"{name}-api" if name == "etid2" and args.api else name
        if key == "etid2-api":
            # Same bank, same code: the non-ridge arms must hold the band. The ridge
            # arms moved from float32 to float64 and are reported, not judged.
            bank = [{k: r[k] for k in r if k in API_BANK_FIELDS} for r in rows]
            ridge = {k: max(abs(r[k]) for r in rows) for k in API_RIDGE_FIELDS}
            v = verdict(name, bank, BANDS[name])
            v["ridge_arms_max_abs_delta_reported"] = ridge
            report[key] = {"verdict": v, "rows": rows}
        else:
            report[key] = {"verdict": verdict(name, rows, BANDS[name]), "rows": rows}
        print(f"== {key}: {report[key]['verdict']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(out.read_text()) if out.exists() else {}
    previous.update(report)
    previous["_seconds_last_run"] = time.time() - started
    out.write_text(json.dumps(previous, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
