"""
AC3: address-space ablation - fixed retrieval vs the learned evidence address.

    docs/AC3_ADDRESS_SPACE_PREREG.md

Evaluation-only re-scoring ablation on the pinned C0 cells. Routing is inference-only
in the contract (the loss never routes), so both addresses see the same trained
weights and no weight is retrained for the comparison:

    bilinear      the pinned address: s_j = <normalize(Pz), normalize(W_j E_j z))>
    fixed_proto   the E0 rule: s_j = max_{c in task j} <normalize(z), normalize(p_c)>

Reference cells: E0 = S11's `L3_per_task`, re-run in-study so the fixed-address
comparison is measured, not quoted.

Usage:
    LD_LIBRARY_PATH=/run/opengl-driver/lib \
        .venv/bin/python experiments/ac3_address_space.py --device cuda
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import e2_evidence as e2  # noqa: E402
import s10_scaling  # noqa: E402
import s11_confirmatory as s11  # noqa: E402
from pal_moe.arch.routers import PrototypeRouter  # noqa: E402

REGIMES = ["coherent", "dispersed"]
T = 20
AGG = "results/agg/aggregation_study.json"
# Reproduction bands, from the development smoke (see the pre-registration).
E2_ANCHOR_BAND = 0.0  # same substrate: the stored C0 cells must be bitwise
E0_REFERENCE_BAND = 1e-6  # the aggregation WTA cells: same rule, float32 noise only
CELL_CEILING_S = 3600


def harness_revision():
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


def train_c0(regime, seed, args, device):
    """The pinned C0 cell: one construction after set_seed, exactly as the chain."""
    _, source = s11.s2_ladder.load_tasks(e2.SOURCE_CACHE)
    tasks = s11.s6b_difficulty.build_construction(source, regime)[:T]
    cell_args = argparse.Namespace(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        rank=e2.OPERATING["rank"],
        evidence_lambda=1.0,
        w_alignment="current",
        p_alignment="plastic",
        seed=seed,
    )
    s11.s2_ladder.set_seed(seed)
    model = e2.E2Model(768, 100, cell_args, device)
    model.train(tasks, seed)
    return model, tasks


def train_e0(regime, seed, args, device):
    """The reference cell: S11 `L3_per_task`, the fixed-address ladder model."""
    _, source = s11.s2_ladder.load_tasks(e2.SOURCE_CACHE)
    tasks = s11.s6b_difficulty.build_construction(source, regime)[:T]
    cell = {
        "construct": regime,
        "level": "L3_per_task",
        "rank": e2.OPERATING["rank"],
        "protos": e2.OPERATING["protos"],
        "top_k": e2.OPERATING["top_k"],
        "num_tasks": e2.OPERATING["num_tasks"],
        "seed": seed,
    }
    cell_args = argparse.Namespace(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_func=1.0,
        seed=seed,
        rank=e2.OPERATING["rank"],
        max_experts=20,
    )
    model = s11.train_model("L3_per_task", tasks, cell, cell_args, device)
    metrics = s11.evaluate(model, "L3_per_task", tasks)
    routing = s10_scaling.routing_report(model, tasks)
    return {
        "accuracy": metrics["accuracy"],
        "coverage_at_1": routing["coverage"]["1"],
        "coverage_at_3": routing["coverage"]["3"],
        "conditional_oracle_at_3": routing["conditional_oracle"]["3"],
        "expert_max_share": routing.get("expert_max_share"),
        "expert_entropy_normalized": routing.get("expert_entropy_normalized"),
    }


def build_router(model, tasks, device):
    """The E0 rule over E2's stored prototypes, registered exactly as the ladder does."""
    router = PrototypeRouter(
        dim=768, num_classes=100, num_experts=len(model.experts)
    ).to(device)
    registered = []
    index = 0
    for task_index, task in enumerate(tasks):
        _, labels = task["splits"]["train"]
        for c in task["classes"]:
            mask = labels == c
            if not bool(mask.any()):
                continue
            stored = model.prototypes[index][0]
            router.register_class(int(c), task_index, stored)
            registered.append((int(c), stored))
            index += 1
    assert index == len(model.prototypes), (index, len(model.prototypes))
    return router, registered


def bilinear_scores(model, feats):
    """The pinned evaluation path, copied verbatim from `E2Model.evaluate`."""
    q = model.P(feats.float())
    scores = []
    for j in range(len(model.experts)):
        adapted = model.experts[j].transform(feats)
        h = e2.normalize(model.W[j](adapted))
        scores.append((e2.normalize(q) * h).sum(dim=-1))
    return torch.stack(scores, dim=1)


def prototype_scores(router):
    def score(feats):
        return router.expert_scores(feats)

    return score


@torch.no_grad()
def evaluate_with(model, tasks, score_fn):
    """`E2Model.evaluate` with the address swapped, plus the coverage@1 and
    global-ratio variants the address comparison needs."""
    correct = total = cov_hits = cov_n_total = oracle_correct = 0
    cov1_n = 0
    oracle_cov_hits = 0
    per_task_cov3, per_task_cov1, per_task_acc = [], [], []
    shares, entropies = [], []
    n_experts = len(model.experts)
    for task in tasks:
        feats, labels = task["splits"]["test"]
        feats, labels = feats.to(model.device), labels.to(model.device)
        s = score_fn(feats)
        pick = s.argmax(dim=1)
        adapted = model._gather(feats, pick)
        logits = s11.s2_ladder.mask_unseen(model.readout.predict(adapted), model.seen)
        hit = logits.argmax(dim=1) == labels
        correct += int(hit.sum())
        total += int(labels.numel())
        owner = int(task["task_id"])
        top3 = s.topk(min(3, s.size(1)), dim=1).indices
        covered = (top3 == owner).any(dim=1)
        cov_n_total += int(covered.sum())
        cov_hits += int(hit[covered].sum())
        covered1 = pick == owner
        cov1_n += int(covered1.sum())
        per_task_cov3.append(float(covered.float().mean()))
        per_task_cov1.append(float(covered1.float().mean()))
        per_task_acc.append(float(hit.float().mean()))
        share = torch.bincount(pick, minlength=n_experts).float() / max(pick.numel(), 1)
        shares.append(float(share.max()))
        p = share[share > 0]
        entropies.append(float(-(p * p.log()).sum()))
        h_owner = e2.normalize(model.W[owner](model.experts[owner].transform(feats)))
        logits_o = s11.s2_ladder.mask_unseen(model.readout.predict(h_owner), model.seen)
        oracle_hit = logits_o.argmax(dim=1) == labels
        oracle_correct += int(oracle_hit.sum())
        oracle_cov_hits += int(oracle_hit[covered].sum())
    accuracy = correct / max(total, 1)
    coverage3 = float(np.mean(per_task_cov3))
    conditional_owner = oracle_cov_hits / max(cov_n_total, 1)
    conditional_routed = cov_hits / max(cov_n_total, 1)
    return {
        "accuracy": accuracy,
        "accuracy_covered": cov_hits / max(cov_n_total, 1),
        "accuracy_uncovered": (correct - cov_hits) / max(total - cov_n_total, 1),
        "coverage_at_1": float(np.mean(per_task_cov1)),
        "coverage_at_3": coverage3,
        "coverage_at_1_global": cov1_n / max(total, 1),
        "coverage_at_3_global": cov_n_total / max(total, 1),
        "oracle_accuracy": oracle_correct / max(total, 1),
        # E2's own key name, kept so the evaluator-equivalence check is exact.
        "conditional_oracle_at_3": conditional_routed,
        "conditional_owner_at_3": conditional_owner,
        "selection_gap_at_3": conditional_owner - conditional_routed,
        "ceiling_at_3": coverage3 * cov_hits / max(cov_n_total, 1),
        "expert_max_share": float(np.mean(shares)),
        "expert_entropy_normalized": float(
            np.mean(entropies) / np.log(max(n_experts, 2))
        ),
        "per_task_accuracy": per_task_acc,
    }


EQUIVALENCE_FIELDS = [
    ("accuracy", "accuracy"),
    ("coverage_at_3", "coverage_at_3"),
    ("oracle_accuracy", "oracle_accuracy"),
    ("conditional_oracle_at_3", "conditional_oracle_at_3"),
    ("ceiling_at_3", "ceiling_at_3"),
    ("per_task_accuracy", "per_task_accuracy"),
]


def run_cell(regime, seed, args, device):
    started = time.time()
    model, tasks = train_c0(regime, seed, args, device)
    anchor = model.evaluate(tasks)
    custom_bilinear = evaluate_with(
        model, tasks, lambda feats: bilinear_scores(model, feats)
    )
    router, registered = build_router(model, tasks, device)
    fixed = evaluate_with(model, tasks, prototype_scores(router))
    prototype_bitwise = all(
        torch.equal(router.means[c], stored) for c, stored in registered
    )
    return {
        "bilinear": custom_bilinear,
        "anchor_metrics": anchor,
        "fixed_proto": fixed,
        "prototype_fidelity": prototype_bitwise,
        "router_param_count": int(sum(p.numel() for p in router.parameters())),
        "seconds": time.time() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,1,2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/ac3")
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "ac3_address_space_study.json")
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]

    study = {
        "schema_version": "1.0",
        "study": "ac3_address_space",
        "prereg": "docs/AC3_ADDRESS_SPACE_PREREG.md",
        "git_revision": harness_revision(),
        "device": args.device,
        "seeds": seeds,
        "cells": [],
        "reference_cells": [],
        "vetoes": {},
        "stats": {},
    }

    def save():
        json.dump(study, open(path, "w"), indent=1)

    for regime in REGIMES:
        for seed in seeds:
            result = run_cell(regime, seed, args, device)
            study["cells"].append({"regime": regime, "seed": seed, **result})
            save()
            b, f = result["bilinear"], result["fixed_proto"]
            print(
                f"[AC3] {regime:10s} seed={seed:<3d} "
                f"bilinear acc={b['accuracy']*100:6.2f} C@3={b['coverage_at_3']:.4f} | "
                f"fixed acc={f['accuracy']*100:6.2f} C@3={f['coverage_at_3']:.4f} "
                f"({result['seconds']:.0f}s)",
                flush=True,
            )
    for regime in REGIMES:
        for seed in seeds:
            e0 = train_e0(regime, seed, args, device)
            study["reference_cells"].append({"regime": regime, "seed": seed, "e0": e0})
            save()
            print(
                f"[AC3] {regime:10s} seed={seed:<3d} reference E0 "
                f"acc={e0['accuracy']*100:6.2f} C@3={e0['coverage_at_3']:.4f}",
                flush=True,
            )

    study["vetoes"] = vetoes(study)
    study["stats"] = stats(study)
    costs = [c["seconds"] for c in study["cells"]]
    study["feasibility"] = {
        "trainings_planned": len(study["cells"]) + len(study["reference_cells"]),
        "mean_cell_seconds": sum(costs) / len(costs) if costs else None,
        "projected_total_hours": (
            (len(study["cells"]) + len(study["reference_cells"]))
            * (sum(costs) / len(costs))
            / 3600
            if costs
            else None
        ),
        "ceiling_hours": CELL_CEILING_S / 3600,
    }
    save()
    print(
        "[AC3] vetoes:",
        json.dumps(
            {
                k: (v["pass"] if isinstance(v, dict) else v)
                for k, v in study["vetoes"].items()
            },
            indent=1,
        ),
        flush=True,
    )
    print(json.dumps(study["stats"], indent=1)[:1600], flush=True)
    print("[AC3] wrote", path)


def vetoes(study):
    cells = study["cells"]
    stored = {}
    for source in ["coupling", "intervention", "owner_side"]:
        for c in json.load(open(f"results/{source}/{source}_study.json"))["cells"]:
            if c["arm"] == "current":
                stored[(source, c["regime"], c["seed"])] = c
    keys = [
        "accuracy",
        "coverage_at_3",
        "conditional_oracle_at_3",
        "ceiling_at_3",
        "oracle_accuracy",
    ]
    anchor_ok, anchor_detail = True, {}
    for cell in cells:
        for source in ["coupling", "intervention", "owner_side"]:
            ref = stored[(source, cell["regime"], cell["seed"])]
            deltas = {
                k: abs(cell["anchor_metrics"][k] - ref[k])
                for k in keys
                if ref.get(k) is not None
            }
            anchor_detail[f"{source}/{cell['regime']}/{cell['seed']}"] = deltas
            if any(v > E2_ANCHOR_BAND for v in deltas.values()):
                anchor_ok = False
    equivalence_ok, equivalence_detail = True, {}
    for cell in cells:
        deltas = {}
        for anchor_key, custom_key in EQUIVALENCE_FIELDS:
            a, b = cell["anchor_metrics"][anchor_key], cell["bilinear"][custom_key]
            if anchor_key == "per_task_accuracy":
                deltas[anchor_key] = max(abs(x - y) for x, y in zip(a, b)) if a else 0.0
            else:
                deltas[anchor_key] = abs(a - b)
        equivalence_detail[f"{cell['regime']}/{cell['seed']}"] = deltas
        if any(v > 0.0 for v in deltas.values()):
            equivalence_ok = False
    ref = {
        (c["regime"], c["seed"]): c
        for c in json.load(open(AGG))["cells"]
        if c["arm"] == "wta"
    }
    identity_ok, identity_detail = True, {}
    reference_ok, reference_detail = True, {}
    for cell in study["reference_cells"]:
        e0, published = cell["e0"], ref[(cell["regime"], cell["seed"])]
        for metric in ["accuracy", "coverage_at_1", "coverage_at_3"]:
            d = abs(e0[metric] - published[metric])
            reference_detail[f"{cell['regime']}/{cell['seed']}/{metric}"] = d
            if d > E0_REFERENCE_BAND:
                reference_ok = False
        fixed = next(
            c
            for c in cells
            if c["regime"] == cell["regime"] and c["seed"] == cell["seed"]
        )["fixed_proto"]
        for metric in ["coverage_at_1", "coverage_at_3"]:
            d = abs(fixed[f"{metric}_global"] - e0[metric])
            identity_detail[f"{cell['regime']}/{cell['seed']}/{metric}"] = d
            if d > 0.0:
                identity_ok = False
    prototype_ok = all(c["prototype_fidelity"] for c in cells)
    paramfree_ok = all(c["router_param_count"] == 0 for c in cells)
    costs = [c["seconds"] for c in cells]
    planned = len(cells) + len(study["reference_cells"])
    mean_cost = sum(costs) / len(costs) if costs else None
    return {
        "bilinear_anchor": {"pass": anchor_ok, "detail": anchor_detail},
        "evaluator_equivalence": {"pass": equivalence_ok, "detail": equivalence_detail},
        "e0_reference": {"pass": reference_ok, "detail": reference_detail},
        "address_identity": {"pass": identity_ok, "detail": identity_detail},
        "prototype_fidelity": {"pass": prototype_ok},
        "param_free_address": {"pass": paramfree_ok},
        "feasibility_ok": bool(
            mean_cost is not None and planned * mean_cost < CELL_CEILING_S
        ),
    }


def stats(study):
    index = {(c["regime"], c["seed"]): c for c in study["cells"]}
    keys = sorted(index)
    out = {}
    tests = {}
    for label, path in [
        ("primary_Acc", "accuracy"),
        ("primary_ceiling", "ceiling_at_3"),
    ]:
        values, per_pair = [], {}
        for regime, seed in keys:
            cell = index[(regime, seed)]
            d = cell["fixed_proto"][path] - cell["bilinear"][path]
            values.append(d)
            per_pair[f"{regime}/{seed}"] = d
        row = s11.paired_stats(values, label)
        row["per_pair"] = per_pair
        out[label] = row
        tests[label] = values
    out["westfall_young"] = s11.westfall_young(tests, len(keys))
    out["equivalence_tost"] = s11.tost([v for v in tests["primary_Acc"]], 0.01)
    return out


if __name__ == "__main__":
    main()
