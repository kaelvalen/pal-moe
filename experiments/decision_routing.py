"""
Decision routing: comparative vs pointwise supervision of the routing scores.

    docs/DECISION_ROUTING_PREREG.md

Both arms train the same `z -> T` linear gate on the L2-normalised feature, with
the same optimizer, epochs, lr, seed, samples and inference (no task id, one
score per expert, argmax). Rows are **not** locked: every step's loss is computed
over all experts seen so far, so no row is trained one-vs-previous (the defect
that sank R2 in the Router Ranking Study). The only difference is the loss:

    pointwise     cross-entropy over the seen experts, target = the owner expert
    comparative   mean over owner-vs-all pairs of
                  max(0, gamma - [s_owner(z) - s_j(z)]),  gamma = 0.2, lambda = 1

Pairs are `(e*, j)` for every other seen expert `j`: no hard-negative mining, no
top-k negatives, no random sampling. The `1/(T_t - 1)` mean keeps the term's
effective weight from growing with the task count.

The expert bank is the ladder's L3, trained once per (regime, seed) and never
touched by either loss. `L4` is router-independent and is read from S11.

Usage:
    python experiments/decision_routing.py --device cuda
    python experiments/decision_routing.py --seeds 42 --report-only
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

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import s10_scaling  # noqa: E402
import s11_confirmatory as s11  # noqa: E402
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402
from rr_factorial import GateRouter  # noqa: E402

SOURCE_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"
REGIMES = ["coherent", "dispersed"]
ARMS = ["pointwise", "comparative"]
OPERATING = {"rank": 8, "protos": 1, "top_k": 1, "num_tasks": 20}
GAMMA = 0.2
LAMBDA = 1.0


def train_gate(arm: str, tasks, dim, num_classes, device, args, seed) -> GateRouter:
    gate = torch.nn.Linear(dim, len(tasks), bias=True).to(device)
    for t, task in enumerate(tasks):
        feats = F.normalize(task["splits"]["train"][0].to(device), dim=-1)
        targets = torch.full((feats.size(0),), t, dtype=torch.long, device=device)
        optimizer = torch.optim.Adam(gate.parameters(), lr=args.gate_lr)
        generator = torch.Generator().manual_seed(seed + t)
        seen = torch.arange(t + 1, device=device)
        for _ in range(args.gate_epochs):
            perm = torch.randperm(feats.size(0), generator=generator)
            for start in range(0, feats.size(0) - args.batch_size + 1, args.batch_size):
                index = perm[start : start + args.batch_size]
                scores = gate(feats[index])
                if arm == "pointwise":
                    loss = F.cross_entropy(scores[:, seen], targets[index])
                else:
                    if t == 0:
                        continue  # no negatives exist yet
                    owner = scores[:, t]
                    others = scores[:, :t]
                    hinge = F.relu(GAMMA - (owner.unsqueeze(1) - others))
                    loss = LAMBDA * hinge.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    gate.eval()
    return GateRouter(gate, len(tasks)).to(device)


def bank_hash(model) -> str:
    digest = hashlib.sha256()
    for tensor in list(model.readout.parameters()) + [
        p for expert in model.experts for p in expert.parameters()
    ]:
        digest.update(tensor.detach().cpu().contiguous().float().numpy().tobytes())
    return digest.hexdigest()[:16]


def bank(regime: str, seed: int, args, device):
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
    return s11.train_model("L3_per_task", tasks, cell, args, device), tasks


@torch.no_grad()
def task_coverages(model, tasks, m="3") -> list[float]:
    out = []
    for task in tasks:
        feats = task["splits"]["test"][0].to(model.device)
        ids, _ = model.router.top_k(feats, k=int(m))
        out.append(float((ids == int(task["task_id"])).any(dim=-1).float().mean()))
    return out


def evaluate(model, tasks) -> dict:
    accuracy = s11.evaluate(model, "L3_per_task", tasks)
    routing = s10_scaling.routing_report(model, tasks)
    coverage = routing.get("coverage", {})
    conditional_oracle = routing.get("conditional_oracle", {})
    per_task = task_coverages(model, tasks, "3")
    return {
        "accuracy": accuracy["accuracy"],
        "acc_matrix": accuracy["acc_matrix"],
        "routing": routing,
        "ceiling": {
            m: coverage[m] * conditional_oracle[m]
            for m in coverage
            if m in conditional_oracle
        },
        "coverage_at_3_per_task": per_task,
        "coverage_at_3_excluding_first": (
            float(sum(per_task[1:]) / len(per_task[1:])) if len(per_task) > 1 else None
        ),
    }


def run_cell(regime, seed, arm, args, device, cache) -> dict:
    key = (regime, seed)
    if key not in cache:
        model, tasks = bank(regime, seed, args, device)
        cache[key] = {"model": model, "tasks": tasks, "hash": bank_hash(model)}
    entry = cache[key]
    model = copy.deepcopy(entry["model"])
    dim = int(entry["tasks"][0]["splits"]["train"][0].size(1))
    gate_router = train_gate(
        arm,
        entry["tasks"],
        dim,
        sum(len(t["classes"]) for t in entry["tasks"]),
        device,
        args,
        seed,
    )
    model.router = gate_router
    result = evaluate(model, entry["tasks"])
    return {
        "regime": regime,
        "seed": seed,
        "arm": arm,
        **result,
        "bank_hash": entry["hash"],
        "num_tasks": len(entry["tasks"]),
    }


def build_hypotheses(cells: list[dict], seeds: list[int]) -> dict:
    index = {(c["regime"], c["arm"], c["seed"]): c for c in cells}

    def get(regime, arm, seed, key, sub=None):
        cell = index.get((regime, arm, seed))
        if not cell:
            return None
        if sub == "coverage":
            return cell["routing"].get("coverage", {}).get(key)
        if sub == "oracle":
            return cell["routing"].get("conditional_oracle", {}).get(key)
        return cell.get(key)

    out: dict = {"effects": {}, "per_arm": {}}
    for regime in REGIMES:
        for label, key, sub in (
            ("delta_C3", "3", "coverage"),
            ("delta_C3_excluding_first", "coverage_at_3_excluding_first", None),
            ("delta_accuracy", "accuracy", None),
            ("delta_conditional_oracle", "3", "oracle"),
        ):
            values = []
            for s in seeds:
                a = get(regime, "comparative", s, key, sub)
                b = get(regime, "pointwise", s, key, sub)
                if a is not None and b is not None:
                    values.append(a - b)
            out["effects"][f"{label}_{regime}"] = s11.paired_stats(
                values, f"comparative - pointwise: {label} ({regime})"
            )
    for regime in REGIMES:
        for arm in ARMS:
            rows = [c for c in cells if c["regime"] == regime and c["arm"] == arm]
            if not rows:
                continue
            out["per_arm"][f"{regime}/{arm}"] = {
                "coverage_at_3": s11.paired_stats(
                    [r["routing"]["coverage"].get("3") for r in rows], "C@3"
                ),
                "coverage_at_3_excluding_first": s11.paired_stats(
                    [r["coverage_at_3_excluding_first"] for r in rows],
                    "C@3 (tasks 1..T-1)",
                ),
                "conditional_oracle_at_3": s11.paired_stats(
                    [r["routing"]["conditional_oracle"].get("3") for r in rows],
                    "conditional oracle@3",
                ),
                "accuracy": s11.paired_stats([r["accuracy"] for r in rows], "accuracy"),
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


def guards(cells: list[dict], seeds: list[int]) -> dict:
    index = {(c["regime"], c["arm"], c["seed"]): c for c in cells}
    same_bank, deltas = {}, {}
    for regime in REGIMES:
        hashes = {
            index[(regime, arm, s)]["bank_hash"]
            for arm in ARMS
            for s in seeds
            if (regime, arm, s) in index
        }
        same_bank[regime] = {"distinct_bank_hashes": len(hashes)}
        reference = s11_lookup(regime)
        for s in seeds:
            cell = index.get((regime, "pointwise", s))
            if cell and s in reference:
                deltas[f"{regime}/{s}"] = abs(reference[s] - cell["accuracy"])
    return {
        "bank_shared_across_arms": same_bank,
        "pointwise_arm_trained": {"checked": len(deltas)},
    }


_S11: dict = {}


def s11_lookup(regime):
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
    return _S11[regime]


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
            "model_family": f"L3_per_task+gate_{cell['arm']}",
            "backbone": "vit_b_16+proj768",
            "backbone_pretraining": "imagenet_frozen",
            "readout": "cosine",
            "expert": "residual_adapter",
        },
        metrics={
            "learning": {
                "accuracy": cell["accuracy"],
                "forgetting": 0.0,
                "acc_matrix": cell["acc_matrix"],
            },
            "cost": {"stored_bytes": 0, "total_params": 0},
        },
        provenance={
            "command": "experiments/decision_routing.py",
            "prereg": "docs/DECISION_ROUTING_PREREG.md",
            "arm": cell["arm"],
            "regime": cell["regime"],
            "gamma": GAMMA,
            "lambda": LAMBDA,
            "bank_hash": cell["bank_hash"],
            "coverage": cell["routing"].get("coverage"),
            "coverage_at_3_per_task": cell["coverage_at_3_per_task"],
            "conditional_oracle": cell["routing"].get("conditional_oracle"),
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
    print("DECISION ROUTING - comparative vs pointwise supervision (paired by seed)")
    print("=" * 100)
    for name, stats in hyps.get("effects", {}).items():
        wy = hyps.get("westfall_young", {}).get(name, {})
        print(f"\n  {name}\n    {stats['label']}\n    per seed: {fmt(stats)}")
        if "sign_p" not in stats:
            print("    (no paired values: the arm never produced this endpoint)")
            continue
        print(
            f"    exact sign p={stats['sign_p']}  permutation p="
            f"{stats['permutation_p']:.4f}  WY p={wy.get('adjusted_p', float('nan')):.4f}"
        )
    print("\nPER ARM")
    for key, arm in hyps.get("per_arm", {}).items():
        print(f"\n  {key}")
        for field, label in (
            ("coverage_at_3", "C@3"),
            ("coverage_at_3_excluding_first", "C@3 (1..T-1)"),
            ("conditional_oracle_at_3", "conditional oracle@3"),
            ("accuracy", "accuracy"),
        ):
            print(f"    {label:20s} {fmt(arm[field])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=ARMS)
    parser.add_argument("--regimes", nargs="*", default=REGIMES)
    parser.add_argument("--seeds", default="42,1,2,3,4,5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument("--gate_epochs", type=int, default=10)
    parser.add_argument("--gate_lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/dr")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    study_path = os.path.join(args.out, "decision_routing_study.json")
    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_func": args.lambda_func,
        "gate_epochs": args.gate_epochs,
        "gate_lr": args.gate_lr,
        "gamma": GAMMA,
        "lambda": LAMBDA,
        "operating": OPERATING,
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
            print(f"[DR] resuming: {len(done)} cells recorded", flush=True)

    def save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "decision_routing",
            "prereg": "docs/DECISION_ROUTING_PREREG.md",
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
        print(f"\n[DR guards] {json.dumps(payload['guards'], indent=1)}")
        return

    grid = [
        {"regime": r, "arm": a, "seed": s}
        for r in args.regimes
        for a in args.arms
        for s in seeds
        if (r, a, s) not in done
    ]
    print(f"[DR] grid: {len(grid)} new cells, seeds={seeds}", flush=True)
    cache: dict = {}
    for cell in grid:
        result = run_cell(
            cell["regime"], cell["seed"], cell["arm"], args, device, cache
        )
        cells.append(result)
        save()
        print(
            f"[DR] {cell['regime']:10s} {cell['arm']:12s} seed={cell['seed']:<3d} "
            f"C@3={(result['routing']['coverage'].get('3') or float('nan')):.4f}  "
            f"C@3(1..T-1)={(result['coverage_at_3_excluding_first'] or float('nan')):.4f}  "
            f"cond_oracle@3="
            f"{(result['routing']['conditional_oracle'].get('3') or float('nan')):.4f}",
            flush=True,
        )
    save()
    payload = json.load(open(study_path))
    print(f"[DR] wrote {study_path}", flush=True)
    _print(payload["hypotheses"])
    print(f"\n[DR guards] {json.dumps(payload['guards'], indent=1)}")


if __name__ == "__main__":
    main()
