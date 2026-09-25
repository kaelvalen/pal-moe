"""
S2 - the complexity ladder under one fixed recipe (docs/STAGE1_PLAN.md).

One backbone, one dataset, one protocol. The only thing that changes between
rows is the model's complexity, and the only thing that changes between columns
is the seed. Nothing is tuned per level: if a rung loses it is reported, and a
sensitivity study would be a separate experiment.

    L0    identity expert   + NCM                     minimum baseline
    L1    identity expert   + ridge / linear          cost of a learned readout
    L2a   1 shared adapter  + readout, JOINT          capacity ceiling (NOT a CL result)
    L2b   1 shared adapter  + readout, sequential     sharing interference
    L3    per-task adapters + readout, prototype router   modular isolation
    L4    per-task adapters + readout, oracle router      routing ceiling
    ceiling_joint_probe      frozen features, joint readout (not CL)

Fixed across every level: the feature cache (byte-identical tensors), the task
partition, the class order, the training budget, the evaluation protocol and
the memory quota. `--seed` is the only axis.

Everything is composed through the S1 registries (`pal_moe.arch`), so the ladder
is a config rather than a branch in a training loop, and every row emits an S0
measurement-contract record.

Usage:
    python experiments/s2_ladder.py --seeds 42 1 2 --device cuda
    python experiments/s2_ladder.py --levels L0_ncm L3_per_task --seeds 42
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from pal_moe.arch import (  # noqa: F401  (re-exported)
    PrototypeRouter,
    build_expert,
    build_readout,
    describe,
    mask_unseen,
    trainable_hook,
)
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks


# Moved to the package in the v3 restructure (phase 1). Re-exported here so every
# runner that does `s2_ladder.X` keeps working unchanged; these are the same objects.
from pal_moe.core.features import iter_batches, load_tasks, set_seed  # noqa: E402,F401
from pal_moe.experts.ladder import (  # noqa: E402,F401
    CLOSED_FORM_READOUTS,
    LADDER,
    LEVELS_BY_NAME,
    LadderModel,
    LevelSpec,
    _entropy,
    forward_transfer,
)

DEFAULT_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"


# ---------------------------------------------------------------------------
# one level, one seed
# ---------------------------------------------------------------------------


def run_level(spec: LevelSpec, tasks, meta, args, device) -> dict:
    set_seed(args.seed)
    dim = int(meta["feature_dim"])
    num_classes = sum(len(t["classes"]) for t in tasks)
    model = LadderModel(spec, dim, num_classes, args, device)

    n = len(tasks)
    R = np.zeros((n, n), dtype=np.float32)
    fwt: list[float] = []
    t0 = time.time()

    if spec.joint:
        all_feats = torch.cat([t["splits"]["train"][0] for t in tasks], dim=0)
        all_labels = torch.cat([t["splits"]["train"][1] for t in tasks], dim=0)
        model.seen = sorted({c for t in tasks for c in t["classes"]})
        for i, task in enumerate(tasks):
            model.register_task(task, i)
        model.fit_task(None, 0, joint_data=(all_feats, all_labels))
        if spec.readout in CLOSED_FORM_READOUTS:
            model.readout.fit(all_feats, all_labels, seen_classes=model.seen)
        for i in range(n):
            R[n - 1, i] = model.evaluate_task(tasks[i])
    else:
        for t, task in enumerate(tasks):
            # Forward transfer: does the representation built so far help the
            # incoming task? Measured with a closed-form probe (see
            # `forward_transfer`). Task 0 has nothing to transfer from.
            if model.seen:
                _adapted, delta = forward_transfer(model, task, num_classes, dim)
                fwt.append(delta)
            model.seen = sorted(set(model.seen) | set(task["classes"]))
            model.fit_task(task, t)
            model.register_task(task, t)
            for i in range(t + 1):
                R[t, i] = model.evaluate_task(tasks[i])

    T = n - 1
    forget = [
        max(0.0, float(np.max(R[i : T + 1, i])) - float(R[T, i])) for i in range(T)
    ]
    routing = model.routing_stats(tasks)
    cost = model.cost()
    return {
        "level": spec.name,
        "seed": args.seed,
        "note": spec.note,
        "joint": spec.joint,
        "avg_accuracy": float(np.mean(R[T, :])),
        "forgetting": float(np.mean(forget)) if forget else 0.0,
        "fwt": float(np.mean(fwt)) if fwt else None,
        "acc_matrix": R.tolist(),
        "oracle_accuracy": float(np.mean(R[T, :])) if spec.router == "oracle" else None,
        "seconds": round(time.time() - t0, 1),
        "latency_ms": measure_latency(model, tasks[0]["splits"]["test"][0], device),
        **routing,
        **cost,
    }




def measure_latency(
    model, feats: torch.Tensor, device, n: int = 128, repeats: int = 20
) -> float:
    """Median per-sample forward latency in milliseconds (best effort)."""
    if device.type != "cuda":
        return float("nan")
    z = feats[:n].to(device)
    with torch.no_grad():
        for _ in range(3):
            model.logits(z, task_id=0)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.logits(z, task_id=0)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0 / z.size(0))
    return float(statistics.median(times))


# ---------------------------------------------------------------------------
# study aggregation
# ---------------------------------------------------------------------------


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "n": 1}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "n": len(values),
    }


def aggregate(records: list[dict], baseline: str = "L0_ncm") -> dict:
    by_level: dict[str, list[dict]] = {}
    for record in records:
        by_level.setdefault(record["level"], []).append(record)

    levels = {
        level: {
            "accuracy": _stats([r["avg_accuracy"] for r in rs]),
            "forgetting": _stats([r["forgetting"] for r in rs]),
            "fwt": _stats([r.get("fwt") for r in rs]),
            "params": rs[0]["params"],
            "flops_forward": rs[0]["flops_forward"],
            "latency_ms": _stats([r.get("latency_ms") for r in rs]),
            "task_recall_at_3": _stats([r.get("task_recall_at_3") for r in rs]),
            "acc_covered": _stats([r.get("acc_covered") for r in rs]),
            "oracle_accuracy": _stats([r.get("oracle_accuracy") for r in rs]),
            "optimizer_steps": rs[0].get("optimizer_steps"),
            "seconds": _stats([r.get("seconds") for r in rs]),
            "seeds": [r["seed"] for r in rs],
        }
        for level, rs in by_level.items()
    }
    # Paired per-seed deltas against the baseline (S0 rule R7).
    base_by_seed = {r["seed"]: r["avg_accuracy"] for r in by_level.get(baseline, [])}
    for level, rs in by_level.items():
        deltas = [
            r["avg_accuracy"] - base_by_seed[r["seed"]]
            for r in rs
            if r["seed"] in base_by_seed
        ]
        levels[level]["delta_vs_baseline"] = _stats(deltas)
        levels[level]["wins"] = sum(1 for d in deltas if d > 0)
        levels[level]["paired_n"] = len(deltas)

    # The routing cost is the L4 - L3 pair, computed per seed.
    if "L3_per_task" in by_level and "L4_oracle" in by_level:
        l3 = {r["seed"]: r["avg_accuracy"] for r in by_level["L3_per_task"]}
        l4 = {r["seed"]: r["avg_accuracy"] for r in by_level["L4_oracle"]}
        gaps = [l4[s] - l3[s] for s in l3 if s in l4]
        levels["routing_cost_L4_minus_L3"] = {"accuracy": _stats(gaps), "n": len(gaps)}
    return {"baseline": baseline, "levels": levels}


# ---------------------------------------------------------------------------
# contract emission
# ---------------------------------------------------------------------------


def _memory_bytes(spec: LevelSpec, meta) -> int:
    dim = int(meta["feature_dim"])
    n_classes = 100
    bytes_ = 0
    if spec.readout == "ncm":
        bytes_ += n_classes * dim * 4
    if spec.router == "prototype":
        bytes_ += n_classes * dim * 4
    return int(bytes_)


def _contract(row, spec: LevelSpec, args, meta, tasks) -> dict:
    latent = meta.get("latent_dim")
    arch = meta["encoder_arch"]
    factors = {
        "dataset": meta["dataset"],
        "protocol": "class_il",
        "task_id_at_inference": spec.router == "oracle",
        "num_tasks": len(tasks),
        "classes_per_task": len(tasks[0]["classes"]),
        "class_order": [list(t["classes"]) for t in tasks],
        "seed": args.seed,
        "model_family": row["level"],
        # The projected backbone is named explicitly: sweeping architectures at
        # a fixed latent dimension is the S3 control, and a record that hides
        # the projection would be unreadable later.
        "backbone": f"{arch}+proj{latent}" if latent else arch,
        "backbone_pretraining": (
            "random"
            if meta.get("encoder_weights") == "none"
            else f"{meta['encoder_weights']}_frozen"
        ),
        "readout": spec.readout,
        "readout_estimator": "offline_mean" if spec.readout == "ncm" else None,
        "expert": spec.expert or "none",
    }
    if latent:
        factors["latent_dim"] = int(latent)
        factors["rep_seed"] = meta.get("rep_seed")
    record = build_run_record(
        factors=factors,
        metrics={
            "learning": {
                "accuracy": row["avg_accuracy"],
                "forgetting": row["forgetting"],
                "fwt": row.get("fwt"),
                "acc_matrix": row["acc_matrix"],
            },
            "cost": {
                "stored_bytes": _memory_bytes(spec, meta),
                "total_params": row["params"],
                "flops_forward": row["flops_forward"],
                "latency_ms": row.get("latency_ms"),
                "optimizer_steps": row["optimizer_steps"],
            },
            "modular": {
                "expert_count": 0 if not spec.expert else len(tasks),
                "task_recall_at_k": (
                    {"3": row["task_recall_at_3"]}
                    if "task_recall_at_3" in row
                    else None
                ),
                "oracle_accuracy": row.get("oracle_accuracy"),
            },
        },
        provenance={
            "git_commit": meta.get("git_commit", "unknown"),
            "device": str(args.device),
            "command": "experiments/s2_ladder.py",
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--seeds", default="42 1 2")
    parser.add_argument("--levels", nargs="*", default=[spec.name for spec in LADDER])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument("--max_experts", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/s2")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    meta, tasks = load_tasks(args.cache)
    meta["git_commit"] = _git_commit()
    print(
        f"[S2] {meta['encoder_arch']}/{meta['encoder_weights']} frozen, "
        f"{len(tasks)} tasks, dim={meta['feature_dim']}, device={device}"
    )
    print(f"[S2] registry: {json.dumps(describe())}")

    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    records, contracts = [], {}
    for name in args.levels:
        spec = LEVELS_BY_NAME[name]
        for seed in seeds:
            args.seed = seed
            row = run_level(spec, tasks, meta, args, device)
            records.append(row)
            contracts[f"{name}__seed{seed}"] = _contract(row, spec, args, meta, tasks)
            print(
                f"[S2] {name:22s} seed={seed:<3d} acc={row['avg_accuracy']*100:6.2f}% "
                f"F={row['forgetting']*100:6.2f}% params={row['params']:8d} "
                f"{row['seconds']:6.1f}s"
            )

    study = aggregate(records)
    os.makedirs(args.out, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "study": "s2_complexity_ladder",
        "factors": {
            "dataset": meta["dataset"],
            "protocol": "class_il",
            "task_id_at_inference": False,
            "backbone": meta["encoder_arch"],
            "backbone_pretraining": f"{meta['encoder_weights']}_frozen",
            "seeds": seeds,
            "fixed": [
                "feature_cache",
                "task_partition",
                "class_order",
                "training_budget",
                "evaluation_protocol",
            ],
        },
        "records": records,
        "contracts": contracts,
        "aggregate": study,
    }
    path = os.path.join(args.out, f"s2_ladder_study{args.tag}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[S2] wrote {path}")
    _print_table(study)


def _print_table(study: dict) -> None:
    print("\n" + "=" * 84)
    print("S2 COMPLEXITY LADDER")
    print("=" * 84)
    print(
        f"{'level':24s} {'acc':>8s} {'std':>7s} {'F':>7s} {'d vs L0':>9s} "
        f"{'params':>9s} {'steps':>7s}"
    )
    for level, s in study["levels"].items():
        if "forgetting" not in s:  # derived rows (e.g. the routing cost)
            continue
        acc = s.get("accuracy") or {}
        d = s.get("delta_vs_baseline") or {}
        print(
            f"{level:24s} {acc.get('mean', float('nan')) * 100:7.2f}% "
            f"{(acc.get('std') or 0) * 100:6.2f}% "
            f"{(s.get('forgetting') or {}).get('mean', float('nan')) * 100:6.2f}% "
            f"{d.get('mean', float('nan')) * 100:+8.2f}% "
            f"{s.get('params', 0):9d} "
            f"{s.get('optimizer_steps', 0):7d}"
        )


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - not a git checkout
        return "unknown"


if __name__ == "__main__":
    main()
