"""E-TID2: a continual class-level ridge router over the unchanged L3 expert bank.

Follows e_tid_ceiling.py: the offline class-level probe (lin_class) beat the
prototype router's C@1 by +7.7 pp (coherent) and +10.4 pp (dispersed). Ridge's
sufficient statistics (A = Z^T Z, B = Z^T Y) are additive, so the offline
class-level solution is reachable under the sequential constraint exactly.

Evaluation-only on the selection, like AC3: the same trained L3_per_task cell,
only the expert id per sample changes.

Arms (same weights):
  proto          the pinned E0 route (PrototypeRouter top-1)             [anchor]
  ridge_routed   expert = owner task of argmax ridge class               [primary]
  ridge_masked   ridge_routed, readout restricted to the routed task's classes
                 (exploratory: a different decoder, S5's masking factor)
  ridge_alone    the continual ridge's own class prediction (== L1_ridge)
  oracle         owner expert given (L4, the ceiling)

Primary contrasts, paired over seeds within a regime:
  P1  ridge_routed - proto        does the better router realise its coverage?
  P2  ridge_routed - ridge_alone  does the bank add anything once routing is fixed?

Readings, fixed before running:
  P2 > 0 in every seed, both regimes   the bank earns value over the best readout
                                        for the first time in the programme
  |P2| < 1 pp                           the bank is redundant given a ridge router
  P2 < 0                                ridge alone wins; the experts dissolve into it
  P1 <= 0 despite higher C@1            selections are not decodable (AC3's gap, again)

Guards (a failure stops the reading, it is not a result):
  G1 evaluator equivalence  the proto arm reproduces model.evaluate_task per task, exactly
  G2 oracle equivalence     the oracle arm reproduces evaluate_task(oracle=True), exactly
  G3 ridge incrementality   continual ridge == one-shot ridge on all train data
                            (argmax identical on every test sample, max|dW| reported)
"""
import argparse, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import s2_ladder, s6b_difficulty, s11_confirmatory as s11  # noqa: E402
from pal_moe.arch import RidgeReadout, mask_unseen  # noqa: E402

CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"
REGIMES = ("coherent", "dispersed")


def fit_ridge(tasks, dim, C, device, ridge):
    cont = RidgeReadout(dim, C, ridge=ridge).to(device)
    for t in tasks:  # sequential, one task at a time
        x, y = t["splits"]["train"]
        cont.fit(x.to(device), y.to(device))
    xs = torch.cat([t["splits"]["train"][0] for t in tasks]).to(device)
    ys = torch.cat([t["splits"]["train"][1] for t in tasks]).to(device)
    once = RidgeReadout(dim, C, ridge=ridge).to(device)
    once.fit(xs, ys)
    return cont, once


@torch.no_grad()
def decode(model, z, ids, allowed=None):
    logits = mask_unseen(model.readout.predict(model.apply_experts(z, ids)), model.seen)
    if allowed is not None:
        logits = logits.masked_fill(~allowed, float("-inf"))
    return logits.argmax(-1)


@torch.no_grad()
def run_cell(regime, seed, args, device, base):
    tasks = s6b_difficulty.build_construction(base, regime)
    T, dim = len(tasks), int(tasks[0]["splits"]["train"][0].size(1))
    C = sum(len(t["classes"]) for t in tasks)
    cell = {"seed": seed, "rank": 8, "protos": 1, "top_k": 1}
    with torch.enable_grad():
        model = s11.train_model("L3_per_task", tasks, cell, args, device)

    cont, once = fit_ridge(tasks, dim, C, device, args.ridge)
    cls2task = torch.zeros(C, dtype=torch.long, device=device)
    task_mask = torch.zeros(T, C, dtype=torch.bool, device=device)
    for t in tasks:
        idx = torch.tensor(t["classes"], device=device)
        cls2task[idx] = t["task_id"]
        task_mask[t["task_id"], idx] = True

    arms = ("proto", "ridge_routed", "ridge_masked", "ridge_alone", "oracle")
    per_task = {a: [] for a in arms}
    cov = {"proto": [0, 0], "ridge": [0, 0]}  # C@1 hits, C@3 hits
    n_total, g3_mismatch, g1, g2 = 0, 0, [], []
    for t in tasks:
        z, y = [v.to(device) for v in t["splits"]["test"]]
        owner = t["task_id"]
        proto_ids = model.router.top_k(z, k=3)[0]
        rlog = cont.predict(z)
        g3_mismatch += int((rlog.argmax(-1) != once.predict(z).argmax(-1)).sum())
        rtask = torch.full((z.size(0), T), -1e9, device=device).scatter_reduce(
            1, cls2task.expand_as(rlog), rlog, "amax")
        ridge_ids = rtask.topk(3, dim=-1).indices
        for name, ids in (("proto", proto_ids), ("ridge", ridge_ids)):
            cov[name][0] += int((ids[:, 0] == owner).sum())
            cov[name][1] += int((ids == owner).any(-1).sum())
        n_total += z.size(0)

        pred = {
            "proto": decode(model, z, proto_ids[:, 0]),
            "ridge_routed": decode(model, z, ridge_ids[:, 0]),
            "ridge_masked": decode(model, z, ridge_ids[:, 0], task_mask[ridge_ids[:, 0]]),
            "ridge_alone": rlog.argmax(-1),
            "oracle": decode(model, z, torch.full_like(y, owner)),
        }
        for a in arms:
            per_task[a].append(float((pred[a] == y).float().mean()))
        g1.append(abs(per_task["proto"][-1] - model.evaluate_task(t, oracle=False, task_id=owner)))
        g2.append(abs(per_task["oracle"][-1] - model.evaluate_task(t, oracle=True, task_id=owner)))

    out = {a: sum(v) / len(v) for a, v in per_task.items()}
    out.update({
        "proto_C@1": cov["proto"][0] / n_total, "proto_C@3": cov["proto"][1] / n_total,
        "ridge_C@1": cov["ridge"][0] / n_total, "ridge_C@3": cov["ridge"][1] / n_total,
        "guards": {
            "G1_max_abs": max(g1), "G2_max_abs": max(g2),
            "G3_argmax_mismatch": g3_mismatch,
            "G3_max_abs_dW": float((cont.W - once.W).abs().max()),
        },
    })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,1,2,3,4,5")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lambda_func", type=float, default=1.0)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/e_tid2/e_tid2_ridge_router.json")
    args = p.parse_args()

    _, base = s2_ladder.load_tasks(CACHE)
    seeds = [int(s) for s in args.seeds.split(",")]
    cells, report = [], {}
    for regime in REGIMES:
        rows = []
        for seed in seeds:
            args.seed = seed
            r = run_cell(regime, seed, args, args.device, base)
            rows.append(r)
            cells.append({"regime": regime, "seed": seed, **r})
            print(regime, seed, {k: round(v, 4) for k, v in r.items() if k != "guards"}, r["guards"])
        report[regime] = {
            "P1_ridge_routed_minus_proto": s11.paired_stats(
                [r["ridge_routed"] - r["proto"] for r in rows], "P1"),
            "P2_ridge_routed_minus_ridge_alone": s11.paired_stats(
                [r["ridge_routed"] - r["ridge_alone"] for r in rows], "P2", sesoi=0.01),
            "exploratory_masked_minus_ridge_alone": s11.paired_stats(
                [r["ridge_masked"] - r["ridge_alone"] for r in rows], "X1"),
        }

    print("\n== primary ==")
    for regime, fam in report.items():
        for name, st in fam.items():
            print(f"{regime:9s} {name:40s} mean {st['mean']:+.4f}  ci95 "
                  f"[{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}]  +{st['positive']}/-{st['negative']}"
                  f"  perm_p {st['permutation_p']:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"args": vars(args), "cells": cells, "report": report}, indent=2))


if __name__ == "__main__":
    main()
