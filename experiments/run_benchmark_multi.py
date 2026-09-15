"""
Multi-seed benchmark driver for PAL-MoE.

Runs experiments/run_benchmark.py once per seed and aggregates mean +/- std
across seeds for every method:

    python experiments/run_benchmark_multi.py --seeds "42 1 2 3 4" --device cuda

Output: <output_dir>/benchmark_multi.json  (per-method means/stds + per-seed)
"""

import os
import sys
import json
import copy
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "experiments" / "run_benchmark.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=str, default="42 1 2 3 4", help="Whitespace-separated seed list"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument(
        "--router_type", type=str, default="dynamic", choices=["dynamic", "distance"]
    )
    parser.add_argument("--lambda_ood", type=float, default=0.0)
    parser.add_argument("--max_proto_drop", type=float, default=None)
    parser.add_argument("--max_proto_acc_drop", type=float, default=None)
    parser.add_argument("--joint_freeze_router", action="store_true", default=False)
    parser.add_argument("--joint_keep_routing_lock", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers forwarded to run_benchmark.py (0 = single-process)",
    )
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--expert_hidden", type=int, default=256)
    parser.add_argument("--conv_channels", type=str, default="32,64,128")
    parser.add_argument("--proto_size", type=int, default=250)
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=None,
        help="Encoder pretraining epochs (default: dataset default)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated method ids (empty = all); forwarded to run_benchmark.py",
    )
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    os.makedirs(args.output_dir, exist_ok=True)

    per_seed = {}
    for s in seeds:
        seed_dir = os.path.join(args.output_dir, f"seed{s}")
        os.makedirs(seed_dir, exist_ok=True)
        cmd = [
            sys.executable,
            str(RUNNER),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--dataset",
            args.dataset,
            "--router_type",
            args.router_type,
            "--seed",
            str(s),
            "--output_dir",
            seed_dir,
            "--lambda_ood",
            str(args.lambda_ood),
            "--feature_dim",
            str(args.feature_dim),
            "--expert_hidden",
            str(args.expert_hidden),
            "--conv_channels",
            args.conv_channels,
            "--proto_size",
            str(args.proto_size),
        ]
        if args.methods:
            cmd += ["--methods", args.methods]
        if args.pretrain_epochs is not None:
            cmd += ["--pretrain_epochs", str(args.pretrain_epochs)]
        if args.joint_keep_routing_lock:
            cmd.append("--joint_keep_routing_lock")
        if args.max_proto_drop is not None:
            cmd += ["--max_proto_drop", str(args.max_proto_drop)]
        if args.max_proto_acc_drop is not None:
            cmd += ["--max_proto_acc_drop", str(args.max_proto_acc_drop)]
        if args.joint_freeze_router:
            cmd.append("--joint_freeze_router")
        if args.num_workers:
            cmd += ["--num_workers", str(args.num_workers)]

        print(f"\n########## Seed {s} ##########")
        subprocess.run(cmd, check=True)

        json_path = os.path.join(seed_dir, f"benchmark_results_seed{s}.json")
        with open(json_path) as f:
            per_seed[s] = json.load(f)

    # Aggregate
    methods = sorted(per_seed[seeds[0]].keys())
    agg = {}
    for m in methods:
        accs = [per_seed[s][m]["acc"] for s in seeds]
        fgts = [per_seed[s][m]["forgetting"] for s in seeds]
        bwts = [per_seed[s][m]["bwt"] for s in seeds]
        experts = [per_seed[s][m]["final_experts"] for s in seeds]
        agg[m] = {
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "forgetting_mean": float(np.mean(fgts)),
            "forgetting_std": float(np.std(fgts)),
            "bwt_mean": float(np.mean(bwts)),
            "bwt_std": float(np.std(bwts)),
            "final_experts": int(round(float(np.mean(experts)))),
        }

    out = {
        "seeds": seeds,
        "config": {
            "epochs": args.epochs,
            "dataset": args.dataset,
            "router_type": args.router_type,
            "lambda_ood": args.lambda_ood,
            "max_proto_drop": args.max_proto_drop,
            "max_proto_acc_drop": args.max_proto_acc_drop,
            "joint_freeze_router": args.joint_freeze_router,
            "joint_keep_routing_lock": args.joint_keep_routing_lock,
            "feature_dim": args.feature_dim,
            "expert_hidden": args.expert_hidden,
            "conv_channels": args.conv_channels,
            "proto_size": args.proto_size,
            "methods": args.methods,
            "pretrain_epochs": args.pretrain_epochs,
        },
        "per_seed": per_seed,
        "aggregated": agg,
    }

    out_path = os.path.join(args.output_dir, "benchmark_multi.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 76)
    print(f"FINAL MULTI-SEED RESULTS ({len(seeds)} seeds, mean +/- std)")
    print("=" * 76)
    header = f"{'Method':45s} {'Avg Acc':>16s} {'Forgetting':>16s} {'Experts':>8s}"
    print(header)
    print("-" * len(header))
    for m, v in agg.items():
        print(
            f"{m:45s} {v['acc_mean'] * 100:6.2f}±{v['acc_std'] * 100:4.2f}% {v['forgetting_mean'] * 100:6.2f}±{v['forgetting_std'] * 100:4.2f}% {v['final_experts']:8d}"
        )
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
