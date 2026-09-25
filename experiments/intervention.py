"""
Intervention: cutting the non-owner evidence path to old projections.

    docs/INTERVENTION_PREREG.md   (V3, Amendment 1)

Arms at T = 20, two regimes, three seeds:
    all          C1, the pinned contract          (re-run, must equal the coupling cell)
    current      C0, old W frozen                 (re-run, must equal the coupling cell)
    owner_only   old W_j see only their own task's prototype terms (the cut)

The cut is a property of the arm, implemented in `E2Model._evidence_loss`: for
`j < t`, `h_j` is detached on prototype rows the expert does not own. The values
are untouched, so every metric of a step is bitwise unchanged; the only manipulated
quantity is the gradient path of old `W_j`.

Instrumentation is passive: loss-form probes in `torch.no_grad()` on a deepcopy,
plus W snapshots. The cut audit runs first and is a veto (V3); the C1/C0 anchor
invariance is checked before the new arm starts (V2).

Usage: python experiments/intervention.py --device cuda
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
COUPLING = "results/coupling/coupling_study.json"
METRIC_KEYS = [
    "accuracy",
    "coverage_at_3",
    "conditional_oracle_at_3",
    "ceiling_at_3",
    "oracle_accuracy",
]


def probe(model, task_index):
    """Loss-form owner/non-owner per-prototype means and sums (prereg section 3).

    No autograd, no `.grad`: everything runs in `torch.no_grad()` on a deepcopy,
    so the training trajectory cannot be touched.
    """
    if task_index == 0 or not model.prototypes:
        return {}
    with torch.no_grad():
        z = torch.stack([p for p, _ in model.prototypes])
        owners = torch.tensor([o for _, o in model.prototypes], device=model.device)
        scores = []
        for j in range(len(model.experts)):
            h = e2.normalize(model.W[j](model.experts[j].transform(z)))
            q = e2.normalize(model.P(z))
            scores.append((q * h).sum(dim=-1))
        ce = F.cross_entropy(torch.stack(scores, dim=1), owners, reduction="none")
    out = {}
    for j in range(task_index):
        own = ce[owners == j]
        non = ce[owners != j]
        out[f"W{j}"] = {
            "owner_mean": float(own.mean()),
            "nonowner_mean": float(non.mean()),
            "owner_sum": float(own.sum()),
            "nonowner_sum": float(non.sum()),
            "n_owner": int(own.numel()),
            "n_nonowner": int(non.numel()),
        }
    return out


def headline(result):
    """asym_pp and the sum-form counterpart at the last task boundary."""
    q = result["probes"].get(f"q{T-1}", {})
    if not q:
        return {}
    n = len(q)
    own_mean = sum(v["owner_mean"] for v in q.values()) / n
    non_mean = sum(v["nonowner_mean"] for v in q.values()) / n
    own_sum = sum(v["owner_sum"] for v in q.values()) / n
    non_sum = sum(v["nonowner_sum"] for v in q.values()) / n
    return {
        "asym_pp": non_mean / max(own_mean, 1e-12),
        "asym_sum": non_sum / max(own_sum, 1e-12),
    }


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
    # the pinned trajectory (the interference study's root cause, fixed there).
    model = e2.E2Model(768, 100, cell_args, device)
    probes, rewrite = {}, {}

    def on_task_start(m, t):
        return {
            "probes": probe(copy.deepcopy(m), t) if t > 0 else None,
            "snaps": {j: m.W[j].weight.detach().clone() for j in range(t + 1)},
        }

    def on_task_end(m, t, st):
        if st["probes"]:
            probes[f"q{t}"] = st["probes"]
        for j in range(t + 1):
            rewrite.setdefault(f"after_t{t}", {})[f"W{j}"] = float(
                (m.W[j].weight.detach() - st["snaps"][j]).norm()
            )

    started = time.time()
    model.train(
        tasks, seed, hooks={"on_task_start": on_task_start, "on_task_end": on_task_end}
    )
    metrics = model.evaluate(tasks)
    result = {
        **metrics,
        "probes": probes,
        "rewrite": rewrite,
        "guard": model.guard,
        "seconds": time.time() - started,
    }
    return {**result, **headline(result)}


def _step(model, t, z, y):
    """One training-batch loss under the model's current arm (audit only)."""
    h = e2.normalize(model.W[t](model.experts[t].transform(z)))
    logits = s11.s2_ladder.mask_unseen(model.readout.predict(h), model.seen)
    l_task = F.cross_entropy(logits, y)
    l_evidence = model._evidence_loss(t)
    return l_task + model.args.evidence_lambda * l_evidence, l_evidence


def audit(device, args, seed):
    """V3, Amendment 1: forward equality, owner-only gradients for old W_j and
    bitwise equality for every other trainable parameter.

    A separate diagnostic run: it trains through all T tasks with the checks at
    every `t > 0`, and its trajectory is not a study cell.
    """
    _, source = s11.s2_ladder.load_tasks(e2.SOURCE_CACHE)
    tasks = s11.s6b_difficulty.build_construction(source, "coherent")[:T]
    cell_args = argparse.Namespace(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        rank=8,
        evidence_lambda=1.0,
        w_alignment=NEW_ARM,
        seed=seed,
    )
    s11.s2_ladder.set_seed(seed)
    model = e2.E2Model(768, 100, cell_args, device)
    model.add_task(tasks[0], 0)
    model.train_task(tasks[0], 0, seed)
    rows = []
    for t in range(1, T):
        model.add_task(tasks[t], t)
        model.seen = sorted(set(model.seen) | set(tasks[t]["classes"]))
        feats, labels = tasks[t]["splits"]["train"]
        feats, labels = feats.to(device), labels.to(device)
        generator = torch.Generator().manual_seed(seed + t)
        batch = next(
            iter(s11.s2_ladder.iter_batches(feats, labels, args.batch_size, generator))
        )
        z, y = batch[0].to(device), batch[1].to(device)

        model.args.w_alignment = "all"
        l_full, _ = _step(model, t, z, y)
        ce_rows, owners = model._evidence_loss(t, rows=True)

        model.args.w_alignment = NEW_ARM
        l_cut, l_ev_cut = _step(model, t, z, y)

        forward_equal = bool(torch.equal(l_full.detach(), l_cut.detach()))

        others = (
            [(f"P.{n}", p) for n, p in model.P.named_parameters()]
            + [(f"g.{n}", p) for n, p in model.readout.named_parameters()]
            + [(f"W{t}", model.W[t].weight)]
            + [(f"E{t}.{n}", p) for n, p in model.experts[t].named_parameters()]
        )
        g_full = torch.autograd.grad(l_full, [p for _, p in others], retain_graph=True)
        g_cut = torch.autograd.grad(l_cut, [p for _, p in others], retain_graph=True)
        deltas = {
            name: float((a.detach() - b.detach()).abs().max())
            for (name, _), a, b in zip(others, g_full, g_cut)
        }
        others_equal = all(delta == 0.0 for delta in deltas.values())

        n_proto = owners.numel()
        rel_errors = []
        exact = True
        for j in range(t):
            mask = (owners == j).float()
            l_ref = (ce_rows * mask).sum() / n_proto
            (g_ref,) = torch.autograd.grad(l_ref, model.W[j].weight, retain_graph=True)
            (g_cut_j,) = torch.autograd.grad(
                l_ev_cut, model.W[j].weight, retain_graph=True
            )
            exact = exact and bool(torch.equal(g_ref.detach(), g_cut_j.detach()))
            denom = float(g_ref.abs().max())
            rel_errors.append(float((g_cut_j - g_ref).abs().max()) / max(denom, 1e-30))

        rows.append(
            {
                "task": t,
                "n_old": t,
                "forward_equal": forward_equal,
                "others_equal": others_equal,
                "others_max_abs_delta": max(deltas.values()),
                "owner_ref_exact": exact,
                "owner_ref_max_rel_error": max(rel_errors) if rel_errors else 0.0,
            }
        )

        model.args.w_alignment = NEW_ARM
        model.train_task(tasks[t], t, seed)

    passed = all(
        r["forward_equal"]
        and r["others_equal"]
        and r["owner_ref_max_rel_error"] <= 1e-6
        for r in rows
    )
    return rows, passed


def check_invariance(cells):
    """V2: every re-run C1/C0 cell must equal the coupling study's cell exactly."""
    reference = {
        (c["arm"], c["regime"], c["seed"]): c
        for c in json.load(open(COUPLING))["cells"]
    }
    detail, ok = {}, True
    for cell in cells:
        if cell["arm"] == NEW_ARM:
            continue
        ref = reference[(cell["arm"], cell["regime"], cell["seed"])]
        deltas = {}
        for key in METRIC_KEYS:
            a, b = cell.get(key), ref.get(key)
            deltas[key] = None if a is None or b is None else abs(a - b)
        detail[f"{cell['arm']}/{cell['regime']}/{cell['seed']}"] = deltas
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
    parser.add_argument("--out", default="results/intervention")
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "intervention_study.json")
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]

    audit_rows, audit_pass = audit(device, args, seed=seeds[0])
    print(f"[IV] cut audit rows={len(audit_rows)} pass={audit_pass}", flush=True)

    study = {
        "schema_version": "1.0",
        "study": "intervention",
        "prereg": "docs/INTERVENTION_PREREG.md",
        "seeds": seeds,
        "cells": [],
        "vetoes": {"cut_audit": {"pass": audit_pass, "rows": audit_rows}},
    }

    def save():
        json.dump(study, open(path, "w"), indent=1)

    if not audit_pass:
        save()
        print("[IV] AUDIT VETO - not executed", flush=True)
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
                    f"[IV] {arm:10s} {regime:10s} seed={seed:<3d} "
                    f"acc={result['accuracy']*100:6.2f} "
                    f"C@3={result['coverage_at_3']:.4f}",
                    flush=True,
                )

    detail, inv_pass = check_invariance(study["cells"])
    study["vetoes"]["anchor_invariance"] = {"pass": inv_pass, "detail": detail}
    save()
    if not inv_pass:
        print("[IV] ANCHOR INVARIANCE VETO - not executed", flush=True)
        return
    print("[IV] anchor invariance PASS", flush=True)

    for regime in REGIMES:
        for seed in seeds:
            result = run_cell(NEW_ARM, regime, seed, args, device)
            study["cells"].append(
                {"regime": regime, "arm": NEW_ARM, "seed": seed, **result}
            )
            save()
            print(
                f"[IV] {NEW_ARM:10s} {regime:10s} seed={seed:<3d} "
                f"acc={result['accuracy']*100:6.2f} "
                f"C@3={result['coverage_at_3']:.4f} "
                f"asym_pp={result.get('asym_pp', float('nan')):.3f} "
                f"asym_sum={result.get('asym_sum', float('nan')):.3f}",
                flush=True,
            )
    print("[IV] wrote", path)


if __name__ == "__main__":
    main()
