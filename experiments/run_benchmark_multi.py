"""
Multi-seed benchmark driver for PAL-MoE.

Runs experiments/run_benchmark.py once per seed and aggregates mean +/- std
across seeds for every method:

    python experiments/run_benchmark_multi.py --seeds "42 1 2 3 4" --device cuda

Output: <output_dir>/benchmark_multi.json  (per-method means/stds + per-seed)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

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
        "--router_type",
        type=str,
        default="dynamic",
        choices=["dynamic", "distance", "attention"],
    )
    parser.add_argument("--lambda_ood", type=float, default=0.0)
    parser.add_argument("--max_proto_drop", type=float, default=None)
    parser.add_argument("--max_proto_acc_drop", type=float, default=None)
    parser.add_argument("--joint_freeze_router", action="store_true", default=False)
    parser.add_argument("--joint_keep_routing_lock", action="store_true")
    parser.add_argument("--lambda_r", type=float, default=0.5)
    parser.add_argument("--lambda_e", type=float, default=2.5)
    parser.add_argument("--stability_every", type=int, default=1)
    parser.add_argument("--ood_every", type=int, default=1)
    parser.add_argument(
        "--buffer_sampling",
        type=str,
        default="recency",
        choices=["recency", "reservoir"],
    )
    parser.add_argument("--ewc_online", action="store_true", default=False)
    parser.add_argument("--max_experts", type=int, default=6)
    parser.add_argument(
        "--proto_routing_threshold",
        type=float,
        default=None,
        help="Max prototype distance for inference anchoring (None = memory threshold)",
    )
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
    parser.add_argument("--freeze_encoder", action="store_true", default=False)
    parser.add_argument("--router_anchor_steps", type=int, default=0)
    parser.add_argument("--router_anchor_lr", type=float, default=1e-3)
    parser.add_argument("--proto_routing_alpha", type=float, default=0.0)
    parser.add_argument("--feature_cache", action="store_true", default=False)
    parser.add_argument("--proto_samples", type=int, default=256)
    parser.add_argument("--proto_threshold", type=str, default="0.5")
    parser.add_argument("--proto_per_class", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--joint_calib_epochs", type=int, default=5)
    parser.add_argument("--refresh_anchors_after_calib", action="store_true")
    parser.add_argument("--keep_optimizer_state", action="store_true")
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated method ids (empty = all); forwarded to run_benchmark.py",
    )
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "JSON config forwarded to every seed (run_benchmark validates it); "
            "when given, only --methods/--seeds/--device/--output_dir are also "
            "forwarded, so the config file defines the recipe"
        ),
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    os.makedirs(args.output_dir, exist_ok=True)

    per_seed = {}
    seeds_meta = {}
    for s in seeds:
        seed_dir = os.path.join(args.output_dir, f"seed{s}")
        os.makedirs(seed_dir, exist_ok=True)
        cmd = [
            sys.executable,
            str(RUNNER),
            "--seed",
            str(s),
            "--device",
            args.device,
            "--output_dir",
            seed_dir,
        ]
        if args.config:
            # Config-driven run: the file defines dataset/epochs/every knob.
            cmd += ["--config", args.config]
            if args.methods:
                cmd += ["--methods", args.methods]
            print(f"\n########## Seed {s} (config={args.config}) ##########")
            subprocess.run(cmd, check=True)
            json_path = os.path.join(seed_dir, f"benchmark_results_seed{s}.json")
            with open(json_path) as f:
                per_seed[s] = json.load(f)
            meta_path = os.path.join(seed_dir, f"benchmark_meta_seed{s}.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    seeds_meta[str(s)] = json.load(f)
            continue
        cmd += [
            "--epochs",
            str(args.epochs),
            "--dataset",
            args.dataset,
            "--router_type",
            args.router_type,
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
        if args.freeze_encoder:
            cmd.append("--freeze_encoder")
        if args.router_anchor_steps:
            cmd += ["--router_anchor_steps", str(args.router_anchor_steps)]
        if args.router_anchor_lr != 1e-3:
            cmd += ["--router_anchor_lr", str(args.router_anchor_lr)]
        if args.proto_routing_alpha > 0:
            cmd += ["--proto_routing_alpha", str(args.proto_routing_alpha)]
        if args.feature_cache:
            cmd.append("--feature_cache")
        if args.proto_samples != 256:
            cmd += ["--proto_samples", str(args.proto_samples)]
        if args.proto_threshold != "0.5":
            cmd += ["--proto_threshold", str(args.proto_threshold)]
        if args.proto_per_class is not None:
            cmd += ["--proto_per_class", str(args.proto_per_class)]
        if args.top_k != 1:
            cmd += ["--top_k", str(args.top_k)]
        if args.joint_calib_epochs != 5:
            cmd += ["--joint_calib_epochs", str(args.joint_calib_epochs)]
        if args.refresh_anchors_after_calib:
            cmd.append("--refresh_anchors_after_calib")
        if args.keep_optimizer_state:
            cmd.append("--keep_optimizer_state")
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
        if args.lambda_r != 0.5:
            cmd += ["--lambda_r", str(args.lambda_r)]
        if args.lambda_e != 2.5:
            cmd += ["--lambda_e", str(args.lambda_e)]
        if args.stability_every != 1:
            cmd += ["--stability_every", str(args.stability_every)]
        if args.ood_every != 1:
            cmd += ["--ood_every", str(args.ood_every)]
        if args.buffer_sampling != "recency":
            cmd += ["--buffer_sampling", args.buffer_sampling]
        if args.ewc_online:
            cmd.append("--ewc_online")
        if args.max_experts != 6:
            cmd += ["--max_experts", str(args.max_experts)]
        if args.proto_routing_threshold is not None:
            cmd += [
                "--proto_routing_threshold",
                str(args.proto_routing_threshold),
            ]

        print(f"\n########## Seed {s} ##########")
        subprocess.run(cmd, check=True)

        json_path = os.path.join(seed_dir, f"benchmark_results_seed{s}.json")
        with open(json_path) as f:
            per_seed[s] = json.load(f)

        meta_path = os.path.join(seed_dir, f"benchmark_meta_seed{s}.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                seeds_meta[str(s)] = json.load(f)

    # Aggregate
    methods = sorted(per_seed[seeds[0]].keys())
    agg = {}
    for m in methods:
        accs = [per_seed[s][m]["acc"] for s in seeds]
        fgts = [per_seed[s][m]["forgetting"] for s in seeds]
        bwts = [per_seed[s][m]["bwt"] for s in seeds]
        experts = [per_seed[s][m]["final_experts"] for s in seeds]

        # Optional metrics are NaN for methods without routing/expert statistics
        # (naive, EWC, replay, ...); nan-aware aggregation keeps them out of the
        # way instead of poisoning the mean.
        def _opt(method, key):
            vals = [
                (
                    float("nan")
                    if per_seed[s][method].get(key) is None
                    else per_seed[s][method][key]
                )
                for s in seeds
            ]
            if all(np.isnan(v) for v in vals):
                return float("nan"), float("nan")
            return float(np.nanmean(vals)), float(np.nanstd(vals))

        kl_mean, kl_std = _opt(m, "router_stability_kl")
        mi_mean, mi_std = _opt(m, "specialization_mi")
        util_mean, util_std = _opt(m, "utilization")
        agg[m] = {
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "forgetting_mean": float(np.mean(fgts)),
            "forgetting_std": float(np.std(fgts)),
            "bwt_mean": float(np.mean(bwts)),
            "bwt_std": float(np.std(bwts)),
            "router_stability_kl_mean": kl_mean,
            "router_stability_kl_std": kl_std,
            "specialization_mi_mean": mi_mean,
            "specialization_mi_std": mi_std,
            "utilization_mean": util_mean,
            "utilization_std": util_std,
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
            "freeze_encoder": args.freeze_encoder,
            "router_anchor_steps": args.router_anchor_steps,
            "router_anchor_lr": args.router_anchor_lr,
            "proto_routing_alpha": args.proto_routing_alpha,
            "feature_cache": args.feature_cache,
            "proto_samples": args.proto_samples,
            "proto_threshold": args.proto_threshold,
            "proto_per_class": args.proto_per_class,
            "top_k": args.top_k,
            "joint_calib_epochs": args.joint_calib_epochs,
            "refresh_anchors_after_calib": args.refresh_anchors_after_calib,
            "keep_optimizer_state": args.keep_optimizer_state,
            "lambda_r": args.lambda_r,
            "lambda_e": args.lambda_e,
            "stability_every": args.stability_every,
            "ood_every": args.ood_every,
            "buffer_sampling": args.buffer_sampling,
            "ewc_online": args.ewc_online,
            "max_experts": args.max_experts,
            "proto_routing_threshold": args.proto_routing_threshold,
        },
        "per_seed": per_seed,
        "aggregated": agg,
        "seeds_meta": seeds_meta,
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
