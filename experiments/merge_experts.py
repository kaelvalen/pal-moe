"""
Merges the task experts of a PAL-MoE checkpoint into a single expert.

Production serving wants one bounded model: this tool loads a per-task
checkpoint, merges every task expert with the chosen policy (parameter soup,
TIES or task arithmetic) and writes the merged expert state dict plus metadata.
The prototype memory is untouched; the merged expert can be wired as a single
serving head or used to initialise the next generation of experts.

Usage:
    python experiments/merge_experts.py --checkpoint results/.../task_4.pt \
        --method ties --output /tmp/merged_expert.pt --dataset mnist
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from experiments.diagnose_checkpoint import build_model, infer_config

from pal_moe.merge import model_soup, task_arithmetic, ties_merge

MERGE_METHODS = {
    "soup": model_soup,
    "ties": ties_merge,
    "task_arithmetic": task_arithmetic,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output", default=None, help="Output path (default: <checkpoint>.merged.pt)"
    )
    parser.add_argument("--method", choices=sorted(MERGE_METHODS), default="soup")
    parser.add_argument("--top_k", type=float, default=0.2, help="TIES trim fraction")
    parser.add_argument(
        "--scaling", type=float, default=1.0, help="Task arithmetic scale"
    )
    parser.add_argument(
        "--dataset", default="mnist", choices=["mnist", "cifar10", "cifar100"]
    )
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = infer_config(payload["model_state"], args.dataset)
    model = build_model(cfg, torch.device("cpu"))
    model.load_state_dict(payload["model_state"])

    if model.num_experts < 2:
        raise SystemExit("[merge] checkpoint has a single expert; nothing to merge")

    merge_fn = MERGE_METHODS[args.method]
    if args.method == "ties":
        merged = merge_fn(model.experts, top_k=args.top_k)
    elif args.method == "task_arithmetic":
        merged = merge_fn(model.experts, scaling=args.scaling)
    else:
        merged = merge_fn(model.experts)

    output = args.output or f"{args.checkpoint}.merged.pt"
    torch.save(
        {
            "expert_state": {k: v.cpu() for k, v in merged.state_dict().items()},
            "meta": {
                "method": args.method,
                "num_experts_merged": model.num_experts,
                "feature_dim": cfg["feature_dim"],
                "expert_hidden": cfg["expert_hidden"],
                "num_classes": cfg["num_classes"],
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            },
        },
        output,
    )
    print(
        f"[merge] merged {model.num_experts} experts with {args.method} -> {output} "
        f"(hidden={merged.hidden_dim}, classes={merged.num_classes})"
    )
    print(json.dumps({"output": output, "method": args.method}, indent=2))


if __name__ == "__main__":
    main()
