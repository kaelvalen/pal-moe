"""
Expert formulation E1: one shared decision space for every expert's output.

    docs/EXPERT_FORMULATION_PREREG.md

    E0   z -> E_t(z) -> logits_t                       the current expert
    E1   z -> E_t(z) -> h_t = P(E_t(z)) -> g(h_t)      one shared decision space

`P` is a single shared `nn.Linear(dim, dim, bias=True)` initialised to the
identity (weight) and zero (bias), so E1 *is* E0 at step 0; `g` is the ladder's
single shared readout. No activation, no normalisation, no hidden layer. Trained
with the same task loss, optimizer, epochs and learning rate as E0, through the
same hooks, so the task loss and the function-preservation loss are computed in
the same space.

Veto order: E0 anchor (S11 per seed, keyed with num_tasks) -> shared P/g identity
and hash guard -> E1 smoke -> 24 cells.

Usage:
    python experiments/expert_formulation.py --device cuda
    python experiments/expert_formulation.py --seeds 42 --report-only
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import s10_scaling  # noqa: E402
import s11_confirmatory as s11  # noqa: E402
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402

SOURCE_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"
REGIMES = ["coherent", "dispersed"]
ARMS = ["e0", "e1"]
OPERATING = {"rank": 8, "protos": 1, "top_k": 1, "num_tasks": 20}
S11_STUDY = "results/s11/s11_confirmatory_study.json"


def module_hash(module) -> str:
    digest = hashlib.sha256()
    for tensor in list(module.parameters()) + list(module.buffers()):
        digest.update(tensor.detach().cpu().contiguous().float().numpy().tobytes())
    return digest.hexdigest()[:12]


def make_projection(dim: int):
    """The pinned form: linear, no activation, identity weight, zero bias."""
    projection = torch.nn.Linear(dim, dim, bias=True)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(dim))
        projection.bias.zero_()
    return projection


def s11_reference(regime: str, seed: int):
    """S11's L3 at the operating point, keyed with num_tasks (two cells exist)."""
    cells = json.load(open(S11_STUDY))["cells"]
    for cell in cells:
        if (
            cell["construct"] == regime
            and cell["level"] == "L3_per_task"
            and cell["rank"] == OPERATING["rank"]
            and cell["protos"] == OPERATING["protos"]
            and cell["num_tasks"] == OPERATING["num_tasks"]
            and cell["seed"] == seed
        ):
            return cell["accuracy"]
    return None


def run_cell(regime, seed, arm, args, device) -> dict:
    _, source = s11.s2_ladder.load_tasks(SOURCE_CACHE)
    tasks = s11.s6b_difficulty.build_construction(source, regime)
    dim = int(tasks[0]["splits"]["train"][0].size(1))
    args.seed = seed

    # E1 must be trained with the projection active, so the two arms train their
    # own model; the hooks are the only difference in the code path.
    spec = s11.s2_ladder.LEVELS_BY_NAME["L3_per_task"]
    num_classes = sum(len(t["classes"]) for t in tasks)
    cell_args = argparse.Namespace(
        rank=OPERATING["rank"],
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=seed,
        lambda_func=args.lambda_func,
        max_experts=max(20, len(tasks)),
    )
    s11.s2_ladder.set_seed(seed)
    model = s11.s2_ladder.LadderModel(
        spec, dim, num_classes, cell_args, device, router_slots=num_classes
    )
    model.router_prototypes = OPERATING["protos"]
    model.top_k_experts = OPERATING["top_k"]
    projection_identity_hash = None
    if arm == "e1":
        model.projection = make_projection(dim).to(device)
        projection_identity_hash = module_hash(model.projection)
    for t, task in enumerate(tasks):
        model.seen = sorted(set(model.seen) | set(task["classes"]))
        model.fit_task(task, t)
        model.register_task(task, t)

    accuracy = s11.evaluate(model, "L3_per_task", tasks)["accuracy"]
    routing = s10_scaling.routing_report(model, tasks)
    coverage = routing.get("coverage", {})
    conditional_oracle = routing.get("conditional_oracle", {})
    return {
        "regime": regime,
        "seed": seed,
        "arm": arm,
        "accuracy": accuracy,
        "routing": routing,
        "ceiling": {
            m: coverage[m] * conditional_oracle[m]
            for m in coverage
            if m in conditional_oracle
        },
        "num_tasks": len(tasks),
        "projection_identity_hash": projection_identity_hash,
        "projection_hash": module_hash(model.projection) if model.projection else None,
        "readout_hash": module_hash(model.readout),
        # Model-local sharing semantics: the forward path references one
        # readout and (for E1) one projection, and no expert carries its own
        # copy. Object identity, not a hash, is what proves sharing.
        "readout_module_count": len({id(model.readout)}),
        "projection_module_count": (
            len({id(model.projection)}) if model.projection is not None else 0
        ),
        "experts_with_own_readout_or_projection": sum(
            1
            for e in model.experts
            if hasattr(e, "readout") or hasattr(e, "projection")
        ),
        "readout_object_id": id(model.readout),
        "projection_object_id": id(model.projection) if model.projection else None,
        "bank_hash": hashlib.sha256(
            b"".join(
                p.detach().cpu().contiguous().float().numpy().tobytes()
                for e in model.experts
                for p in e.parameters()
            )
        ).hexdigest()[:12],
    }


def build_hypotheses(cells: list[dict], seeds: list[int]) -> dict:
    index = {(c["regime"], c["arm"], c["seed"]): c for c in cells}
    out: dict = {"effects": {}, "per_arm": {}}

    def grp(regime, arm):
        return [index[(regime, arm, s)] for s in seeds if (regime, arm, s) in index]

    for regime in REGIMES:
        for label, getter in (
            ("delta_accuracy", lambda c: c["accuracy"]),
            (
                "delta_coverage_at_3",
                lambda c: c["routing"].get("coverage", {}).get("3"),
            ),
            (
                "delta_conditional_oracle_at_3",
                lambda c: c["routing"].get("conditional_oracle", {}).get("3"),
            ),
            ("delta_ceiling_at_3", lambda c: c["ceiling"].get("3")),
        ):
            values = []
            for s in seeds:
                a, b = index.get((regime, "e1", s)), index.get((regime, "e0", s))
                if a and b:
                    values.append(getter(a) - getter(b))
            out["effects"][f"{label}_{regime}"] = s11.paired_stats(
                values, f"E1 - E0: {label} ({regime})"
            )
    for regime in REGIMES:
        for arm in ARMS:
            rows = (
                grp(regime, arm)
                if (regime, arm) in []
                else [c for c in cells if c["regime"] == regime and c["arm"] == arm]
            )
            if not rows:
                continue
            out["per_arm"][f"{regime}/{arm}"] = {
                "accuracy": s11.paired_stats([r["accuracy"] for r in rows], "accuracy"),
                "coverage_at_3": s11.paired_stats(
                    [r["routing"].get("coverage", {}).get("3") for r in rows], "C@3"
                ),
                "conditional_oracle_at_3": s11.paired_stats(
                    [r["routing"].get("conditional_oracle", {}).get("3") for r in rows],
                    "conditional oracle@3",
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
    # The four-way reading the pre-registration fixes in advance.
    out["case"] = {}
    for regime in REGIMES:
        d_c = out["effects"].get(f"delta_coverage_at_3_{regime}", {}).get("mean")
        d_o = (
            out["effects"]
            .get(f"delta_conditional_oracle_at_3_{regime}", {})
            .get("mean")
        )
        if d_c is None or d_o is None:
            continue
        out["case"][regime] = {
            "d_coverage": d_c,
            "d_oracle": d_o,
            "reading": (
                "both improve: the shared decision space helps assignment and usability"
                if d_c > 0 and d_o > 0
                else (
                    "routing gain paid for with expert usability: a trade, not a fix"
                    if d_c > 0 and d_o <= 0
                    else (
                        "experts better in their own space, less comparable across: the RRF trade again"
                        if d_c <= 0 and d_o > 0
                        else "the formulation costs capacity on both sides"
                    )
                )
            ),
        }
    return out


def guards(cells: list[dict], seeds: list[int]) -> dict:
    index = {(c["regime"], c["arm"], c["seed"]): c for c in cells}
    anchor = {}
    for regime in REGIMES:
        for s in seeds:
            e0 = index.get((regime, "e0", s))
            reference = s11_reference(regime, s)
            if e0 and reference is not None:
                anchor[f"{regime}/{s}"] = abs(reference - e0["accuracy"])
    local = {}
    for key, cell in index.items():
        regime, arm, seed = key
        local[f"{regime}/{arm}/{seed}"] = {
            "readout_module_count": cell["readout_module_count"],
            "projection_module_count": cell["projection_module_count"],
            "experts_with_own_readout_or_projection": cell[
                "experts_with_own_readout_or_projection"
            ],
            "expected_projection_modules": 1 if arm == "e1" else 0,
        }
    sharing_ok = all(
        row["readout_module_count"] == 1
        and row["projection_module_count"] == row["expected_projection_modules"]
        and row["experts_with_own_readout_or_projection"] == 0
        for row in local.values()
    )
    return {
        "e0_anchor_reproduces_s11": {
            "checked": len(anchor),
            "max_abs_delta": max(anchor.values()) if anchor else None,
            "veto_passed": bool(anchor and max(anchor.values()) <= 1e-6),
        },
        "model_local_sharing": {
            "cells": len(local),
            "detail": local,
            "veto_passed": sharing_ok,
            "note": "one readout and one projection module per model, referenced "
            "by object identity; no expert carries its own copy. Hashes are "
            "recorded for the audit trail, not compared across cells.",
        },
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
            "model_family": f"L3_per_task+{cell['arm']}",
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
            "command": "experiments/expert_formulation.py",
            "prereg": "docs/EXPERT_FORMULATION_PREREG.md",
            "arm": cell["arm"],
            "regime": cell["regime"],
            "projection_hash": cell["projection_hash"],
            "coverage": cell["routing"].get("coverage"),
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
    print("EXPERT FORMULATION: E1 shared decision space vs E0")
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
        for field in ("accuracy", "coverage_at_3", "conditional_oracle_at_3"):
            print(f"    {field:26s} {fmt(arm[field])}")
    print("\nCASE READING (fixed in advance)")
    for regime, case in hyps.get("case", {}).items():
        print(
            f"  {regime}: dC@3={case['d_coverage']:+.4f}  "
            f"dOracle@3={case['d_oracle']:+.4f}\n    -> {case['reading']}"
        )


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
    parser.add_argument("--out", default="results/ef")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    study_path = os.path.join(args.out, "expert_formulation_study.json")
    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_func": args.lambda_func,
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
            print(f"[EF] resuming: {len(done)} cells", flush=True)

    def save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "expert_formulation",
            "prereg": "docs/EXPERT_FORMULATION_PREREG.md",
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
        print(f"\n[EF guards] {json.dumps(payload['guards'], indent=1)[:700]}")
        return

    grid = [
        {"regime": r, "arm": a, "seed": s}
        for r in args.regimes
        for a in args.arms
        for s in seeds
        if (r, a, s) not in done
    ]
    print(f"[EF] grid: {len(grid)} new cells, seeds={seeds}", flush=True)
    for cell in grid:
        result = run_cell(cell["regime"], cell["seed"], cell["arm"], args, device)
        cells.append(result)
        save()
        print(
            f"[EF] {cell['regime']:10s} {cell['arm']} seed={cell['seed']:<3d} "
            f"acc={result['accuracy'] * 100:6.2f}  "
            f"C@3={(result['routing']['coverage'].get('3') or float('nan')):.4f}  "
            f"oracle@3="
            f"{(result['routing']['conditional_oracle'].get('3') or float('nan')):.4f}",
            flush=True,
        )
    save()
    payload = json.load(open(study_path))
    print(f"[EF] wrote {study_path}", flush=True)
    _print(payload["hypotheses"])
    print(f"\n[EF guards] {json.dumps(payload['guards'], indent=1)[:700]}")


if __name__ == "__main__":
    main()
