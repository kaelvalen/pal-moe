"""
Decision rule: winner-take-all vs uniform top-3 aggregation.

    docs/AGGREGATION_PREREG.md

Same router, same scores, same candidate set; both arms compute the same three
experts' posteriors and differ only in how they are combined:

    WTA       use the top-ranked candidate's posterior
    mixture   uniform mean of the three candidates' posteriors

`covered` is defined by the OWNER expert being in C_3, not by L4 and not by the
WTA pick. Primary is `Delta Acc = Acc_mixture - Acc_WTA`; coverage cannot move and
is a guard.

Usage:
    python experiments/aggregation.py --device cuda
    python experiments/aggregation.py --seeds 42 --report-only
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import s11_confirmatory as s11  # noqa: E402
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402

SOURCE_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"
REGIMES = ["coherent", "dispersed"]
ARMS = ["wta", "mixture"]
OPERATING = {"rank": 8, "protos": 1, "top_k": 1, "num_tasks": 20}
TOP_M = 3


def bank_hash(model) -> str:
    digest = hashlib.sha256()
    for tensor in list(model.readout.parameters()) + [
        p for expert in model.experts for p in expert.parameters()
    ]:
        digest.update(tensor.detach().cpu().contiguous().float().numpy().tobytes())
    return digest.hexdigest()[:12]


@torch.no_grad()
def arm_metrics(model, tasks, arm: str) -> dict:
    """Both arms from one forward: the same three experts, combined differently."""
    correct = total = covered_n = covered_correct = 0
    coverage_correct = 0
    posterior_hash = hashlib.sha256()
    per_task_coverage = []
    for task in tasks:
        feats, labels = task["splits"]["test"]
        feats, labels = feats.to(model.device), labels.to(model.device)
        owner = int(task["task_id"])
        scores = model.router.expert_scores(feats)
        values, ids = scores.topk(TOP_M, dim=-1)
        posterior_hash.update(values.detach().cpu().numpy().tobytes())
        # The three candidates' posteriors over the FULL global class space,
        # from the unchanged ladder readout and softmax. No expert-local class
        # masking in either arm: masking would re-introduce the task/class
        # masking factor S5 separated out, and would make the WTA arm differ from
        # the ladder's own path (the anchor would fail by construction).
        blocks = []
        for j in range(TOP_M):
            e = ids[:, j]
            logits_e = model.readout.predict(model.apply_experts(feats, e))
            probs = F.softmax(logits_e, dim=-1)
            blocks.append((e, probs))
            posterior_hash.update(probs.detach().cpu().numpy().tobytes())
        if arm == "wta":
            chosen = blocks[0][1]
        else:
            chosen = torch.stack([p for _, p in blocks], dim=0).mean(dim=0)
        pred = chosen.argmax(dim=-1)
        hit = pred == labels
        correct += int(hit.sum())
        total += int(labels.numel())
        covered = (ids == owner).any(dim=-1)
        covered_n += int(covered.sum())
        covered_correct += int(hit[covered].sum())
        coverage_correct += int((ids[:, 0] == owner).sum())
        per_task_coverage.append(float((ids == owner).any(dim=-1).float().mean()))
    return {
        "accuracy": correct / max(total, 1),
        "accuracy_covered": covered_correct / max(covered_n, 1),
        "covered_fraction": covered_n / max(total, 1),
        "coverage_at_3": float(np.mean(per_task_coverage)),
        "coverage_at_1": coverage_correct / max(total, 1),
        "posterior_hash": posterior_hash.hexdigest()[:12],
        "posteriors_are_full_class": True,
        "candidate_ids_hash": None,
    }


def run_cell(regime, seed, arm, args, device, cache) -> dict:
    key = (regime, seed)
    if key not in cache:
        _, source = s11.s2_ladder.load_tasks(SOURCE_CACHE)
        tasks = s11.s6b_difficulty.build_construction(source, regime)
        cell = {
            "construct": regime,
            "level": "L3_per_task",
            "rank": OPERATING["rank"],
            "protos": OPERATING["protos"],
            "top_k": OPERATING["top_k"],
            "num_tasks": OPERATING["num_tasks"],
            "seed": seed,
        }
        args.seed = seed
        model = s11.train_model("L3_per_task", tasks, cell, args, device)
        cache[key] = {"model": model, "tasks": tasks, "hash": bank_hash(model)}
    entry = cache[key]
    model = copy.deepcopy(entry["model"])
    metrics = arm_metrics(model, entry["tasks"], arm)
    return {
        "regime": regime,
        "seed": seed,
        "arm": arm,
        **metrics,
        "bank_hash": entry["hash"],
        "num_tasks": len(entry["tasks"]),
    }


def build_hypotheses(cells: list[dict], seeds: list[int]) -> dict:
    index = {(c["regime"], c["arm"], c["seed"]): c for c in cells}
    out: dict = {"effects": {}, "per_arm": {}}

    def pairs(regime, field):
        values = []
        for s in seeds:
            a = index.get((regime, "mixture", s))
            b = index.get((regime, "wta", s))
            if a and b:
                values.append(a[field] - b[field])
        return values

    for regime in REGIMES:
        for field in ("accuracy", "accuracy_covered"):
            out["effects"][f"delta_{field}_{regime}"] = s11.paired_stats(
                pairs(regime, field), f"mixture - wta: {field} ({regime})"
            )
        out["effects"][f"delta_coverage_at_3_{regime}"] = s11.paired_stats(
            pairs(regime, "coverage_at_3"), f"mixture - wta: coverage@3 ({regime})"
        )
    for regime in REGIMES:
        for arm in ARMS:
            rows = [c for c in cells if c["regime"] == regime and c["arm"] == arm]
            if not rows:
                continue
            out["per_arm"][f"{regime}/{arm}"] = {
                "accuracy": s11.paired_stats([r["accuracy"] for r in rows], "accuracy"),
                "accuracy_covered": s11.paired_stats(
                    [r["accuracy_covered"] for r in rows], "accuracy | covered"
                ),
                "coverage_at_3": s11.paired_stats(
                    [r["coverage_at_3"] for r in rows], "coverage@3"
                ),
                "n": len(rows),
            }
    out["westfall_young"] = s11.westfall_young(
        {
            name: stats["per_seed"]
            for name, stats in out["effects"].items()
            if stats.get("n") == len(seeds)
        },
        len(seeds),
    )
    return out


_S11: dict = {}


def s11_accuracy(regime, seed):
    if regime not in _S11:
        cells = json.load(open("results/s11/s11_confirmatory_study.json"))["cells"]
        _S11[regime] = {
            c["seed"]: c["accuracy"]
            for c in cells
            if c["construct"] == regime
            and c["level"] == "L3_per_task"
            and c["rank"] == OPERATING["rank"]
            and c["protos"] == OPERATING["protos"]
            and c["num_tasks"] == OPERATING["num_tasks"]
        }
    return _S11[regime].get(seed)


def guards(cells: list[dict], seeds: list[int]) -> dict:
    """Hierarchy: WTA anchor, then the identity guards."""
    index = {(c["regime"], c["arm"], c["seed"]): c for c in cells}
    anchor, identities, bank = {}, {}, {}
    for regime in REGIMES:
        for s in seeds:
            wta = index.get((regime, "wta", s))
            mix = index.get((regime, "mixture", s))
            reference = s11_accuracy(regime, s)
            if wta and reference is not None:
                anchor[f"{regime}/{s}"] = abs(reference - wta["accuracy"])
            if wta and mix:
                identities[f"{regime}/{s}"] = {
                    "coverage_delta": abs(wta["coverage_at_3"] - mix["coverage_at_3"]),
                    "covered_fraction_delta": abs(
                        wta["covered_fraction"] - mix["covered_fraction"]
                    ),
                    "expert_output_hash_equal": wta["posterior_hash"]
                    == mix["posterior_hash"],
                    "global_class_postervior_guard": bool(
                        wta["posteriors_are_full_class"]
                        and mix["posteriors_are_full_class"]
                    ),
                    "bank_equal": wta["bank_hash"] == mix["bank_hash"],
                }
        hashes = {
            index[(regime, arm, s)]["bank_hash"]
            for arm in ARMS
            for s in seeds
            if (regime, arm, s) in index
        }
        bank[regime] = {"distinct_bank_hashes": len(hashes)}
    worst_anchor = max(anchor.values()) if anchor else None
    identity_ok = all(
        row["coverage_delta"] < 1e-12
        and row["covered_fraction_delta"] < 1e-12
        and row["expert_output_hash_equal"]
        and row["bank_equal"]
        for row in identities.values()
    )
    return {
        "wta_anchor_reproduces_s11": {
            "checked": len(anchor),
            "max_abs_delta": worst_anchor,
            "veto_passed": bool(worst_anchor is not None and worst_anchor <= 1e-6),
        },
        "identity_guards": {
            "cells": len(identities),
            "all_passed": identity_ok,
            "detail": identities,
        },
        "bank_hashes": bank,
    }


def _contract(cell: dict) -> dict:
    record = build_run_record(
        factors={
            "dataset": "cifar100",
            "protocol": "class_il",
            "task_id_at_inference": False,
            "class_masking": False,
            "routing_mode": "learned",
            "increment_type": "class",
            "class_space": "task",
            "num_tasks": cell["num_tasks"],
            "classes_per_task": 5,
            "seed": cell["seed"],
            "task_order_seed": None,
            "model_family": f"L3_per_task+{cell['arm']}_top3",
            "backbone": "vit_b_16+proj768",
            "backbone_pretraining": "imagenet_frozen",
            "readout": "cosine",
            "expert": "residual_adapter",
        },
        metrics={
            "learning": {
                "accuracy": cell["accuracy"],
                "forgetting": 0.0,
                "acc_matrix": [],
            },
            "cost": {"stored_bytes": 0, "total_params": 0},
        },
        provenance={
            "command": "experiments/aggregation.py",
            "prereg": "docs/AGGREGATION_PREREG.md",
            "arm": cell["arm"],
            "regime": cell["regime"],
            "coverage_at_3": cell["coverage_at_3"],
            "accuracy_covered": cell["accuracy_covered"],
            "posterior_hash": cell["posterior_hash"],
            "bank_hash": cell["bank_hash"],
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


def _print(hyps: dict) -> None:
    def fmt(stats):
        if not stats or stats.get("n", 0) == 0 or "per_seed" not in stats:
            return "n/a"
        per = " ".join(f"{v:+.4f}" for v in stats["per_seed"])
        return f"{per}   mean {stats['mean']:+.4f} sd {stats['sd']:.4f}"

    print("\n" + "=" * 100)
    print("DECISION RULE: winner-take-all vs uniform top-3 aggregation")
    print("=" * 100)
    for name, stats in hyps.get("effects", {}).items():
        wy = hyps.get("westfall_young", {}).get(name, {})
        print(f"\n  {name}\n    {stats['label']}\n    per seed: {fmt(stats)}")
        if "sign_p" in stats:
            print(
                f"    exact sign p={stats['sign_p']}  permutation p="
                f"{stats['permutation_p']:.4f}  WY p={wy.get('adjusted_p', float('nan')):.4f}"
            )
    print("\nPER ARM")
    for key, arm in hyps.get("per_arm", {}).items():
        print(f"\n  {key}")
        for field in ("accuracy", "accuracy_covered", "coverage_at_3"):
            print(f"    {field:18s} {fmt(arm[field])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=ARMS)
    parser.add_argument("--regimes", nargs="*", default=REGIMES)
    parser.add_argument("--seeds", default="42,1,2,3,4,5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/agg")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    study_path = os.path.join(args.out, "aggregation_study.json")
    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_func": args.lambda_func,
        "operating": OPERATING,
        "top_m": TOP_M,
        "arms": list(args.arms),
        "regimes": list(args.regimes),
    }
    cells: list[dict] = []
    done: set[tuple] = set()
    if os.path.exists(study_path) and not args.force:
        previous = json.load(open(study_path))
        if previous.get("recipe") == recipe:
            cells = previous.get("cells", [])
            done = {(c["regime"], c["arm"], c["seed"]) for c in cells}
            print(f"[AGG] resuming: {len(done)} cells", flush=True)

    def save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "aggregation",
            "prereg": "docs/AGGREGATION_PREREG.md",
            "backbone": "vit_b_16+proj768",
            "recipe": recipe,
            "seeds": seeds,
            "cells": cells,
            "contracts": {
                f"{c['regime']}__{c['arm']}__seed{c['seed']}": _contract(c)
                for c in cells
            },
            "hypotheses": build_hypotheses(cells, seeds) if cells else {},
            "guards": guards(cells, seeds) if cells else {},
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)

    if args.report_only:
        payload = json.load(open(study_path))
        payload["hypotheses"] = build_hypotheses(payload["cells"], seeds)
        payload["guards"] = guards(payload["cells"], seeds)
        payload["contracts"] = {
            f"{c['regime']}__{c['arm']}__seed{c['seed']}": _contract(c)
            for c in payload["cells"]
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)
        _print(payload["hypotheses"])
        print(f"\n[AGG guards] {json.dumps(payload['guards'], indent=1)[:900]}")
        return

    grid = [
        {"regime": r, "arm": a, "seed": s}
        for r in args.regimes
        for a in args.arms
        for s in seeds
        if (r, a, s) not in done
    ]
    print(f"[AGG] grid: {len(grid)} new cells, seeds={seeds}", flush=True)
    cache: dict = {}
    for cell in grid:
        result = run_cell(
            cell["regime"], cell["seed"], cell["arm"], args, device, cache
        )
        cells.append(result)
        save()
        print(
            f"[AGG] {cell['regime']:10s} {cell['arm']:8s} seed={cell['seed']:<3d} "
            f"acc={result['accuracy'] * 100:6.2f}  acc|covered="
            f"{result['accuracy_covered'] * 100:6.2f}  C@3={result['coverage_at_3']:.4f}",
            flush=True,
        )
    save()
    payload = json.load(open(study_path))
    print(f"[AGG] wrote {study_path}", flush=True)
    _print(payload["hypotheses"])
    print(f"\n[AGG guards] {json.dumps(payload['guards'], indent=1)[:900]}")


if __name__ == "__main__":
    main()
