"""
Per-sample forward latency for the benchmark models.

The benchmark JSONs report `fit_seconds` (training cost) and the three
parameter counts, but not serving latency. This script times the forward pass
of the same model geometries the runner builds:

- single-head baseline: frozen encoder + one MLP head;
- PAL-MoE: frozen encoder + router + N experts (top-1, so one expert is active
  per sample, exactly like the published recipes).

Latency is measured at batch size 1 (interactive) and 128 (throughput), with
CUDA synchronisation and warmup. It is a pure measurement: no training, no
results, no RNG effect on any benchmark run.

Usage:
    python experiments/measure_latency.py --dataset cifar10 --encoder_arch resnet18 \
        --encoder_weights imagenet --feature_dim 256 --expert_hidden 512 \
        --num_experts 6 --batch_sizes 1,128
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from pal_moe.factory import build_encoder, build_moe, build_single_head

DATASET_NORM = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "mnist": ((0.1307,), (0.3081,)),
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _time_forward(
    model: torch.nn.Module,
    input_shape: tuple,
    batch_size: int,
    iters: int,
    warmup: int,
    device: torch.device,
) -> float:
    """Milliseconds per sample for a forward pass at `batch_size`."""
    model.eval()
    x = torch.randn(batch_size, *input_shape, device=device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    _sync(device)
    elapsed = time.perf_counter() - t0
    return elapsed / max(iters, 1) / max(batch_size, 1) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--encoder_arch", type=str, default="resnet18")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--expert_hidden", type=int, default=512)
    parser.add_argument(
        "--num_experts",
        type=str,
        default="6",
        help="Comma-separated expert counts for the PAL-MoE rows",
    )
    parser.add_argument("--batch_sizes", type=str, default="1,128")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--input_size", type=int, default=32)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    arch = args.encoder_arch
    if arch == "mlp":
        input_shape = (1, 784)
    else:
        input_shape = (3, args.input_size, args.input_size)
    num_classes = 100 if args.dataset == "cifar100" else 10
    norm = DATASET_NORM.get(args.dataset)

    encoder = build_encoder(
        input_dim=(
            1 * 784
            if arch == "mlp"
            else input_shape[0] * input_shape[1] * input_shape[2]
        ),
        feature_dim=args.feature_dim,
        arch=arch,
        hidden_dims=(256, 128) if arch == "mlp" else None,
        backbone_weights=args.encoder_weights,
        device=device,
    )
    if norm is not None:
        encoder.input_mean = norm[0]
        encoder.input_std = norm[1]
    encoder.freeze()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    expert_counts = [int(n) for n in args.num_experts.split(",") if n.strip()]

    rows = []

    single = build_single_head(
        encoder, args.feature_dim, args.expert_hidden, num_classes, device
    )
    for bs in batch_sizes:
        rows.append(
            {
                "model": "single head",
                "experts": 1,
                "total_params": int(sum(p.numel() for p in single.parameters())),
                "batch_size": bs,
                "latency_ms_per_sample": round(
                    _time_forward(
                        single, input_shape, bs, args.iters, args.warmup, device
                    ),
                    4,
                ),
            }
        )
        print(
            f"  single head            bs={bs:4d}  {rows[-1]['latency_ms_per_sample']:.4f} ms/sample"
        )

    for n_experts in expert_counts:
        moe = build_moe(
            encoder,
            args.feature_dim,
            args.expert_hidden,
            num_classes,
            num_experts=n_experts,
            router_type="dynamic",
            top_k=1,
            device=device,
        )
        for bs in batch_sizes:
            rows.append(
                {
                    "model": f"PAL-MoE ({n_experts} experts)",
                    "experts": n_experts,
                    "total_params": int(sum(p.numel() for p in moe.parameters())),
                    "batch_size": bs,
                    "latency_ms_per_sample": round(
                        _time_forward(
                            moe, input_shape, bs, args.iters, args.warmup, device
                        ),
                        4,
                    ),
                }
            )
            print(
                f"  PAL-MoE {n_experts:2d} experts     bs={bs:4d}  "
                f"{rows[-1]['latency_ms_per_sample']:.4f} ms/sample"
            )

    if args.output:
        payload = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "encoder_arch": arch,
            "encoder_weights": args.encoder_weights,
            "feature_dim": args.feature_dim,
            "expert_hidden": args.expert_hidden,
            "device": str(device),
            "input_shape": list(input_shape),
            "iters": args.iters,
            "warmup": args.warmup,
            "rows": rows,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved latency results to {args.output}")


if __name__ == "__main__":
    main()
