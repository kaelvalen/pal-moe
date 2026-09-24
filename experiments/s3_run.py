"""
S3 driver, in Python: extract one projected cache per backbone condition, then
run the S2 ladder on each.

The bash version of this driver produced no output at all on this machine
(sourcing the CUDA env under `set -u` swallowed it), so the driver lives in
Python where the failure modes are visible. The ladder code is imported
directly, so S2 and S3 cannot drift apart.

Usage:
    python experiments/s3_run.py --device cuda
    python experiments/s3_run.py --only random,vit_b16_imagenet --seeds 42
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import s2_ladder  # noqa: E402
import s3_backbones  # noqa: E402

LEVELS = [
    "L0_ncm",
    "L1_ridge",
    "L1_cosine",
    "L2a_shared_joint",
    "L2b_shared_seq",
    "L3_per_task",
    "L4_oracle",
]


def run_with_argv(argv: list[str]) -> None:
    """Call a script's `main()` with a synthetic command line."""
    old = sys.argv
    sys.argv = argv
    try:
        s2_ladder.main()
    finally:
        sys.argv = old


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="")
    parser.add_argument("--latent_dim", type=int, default=768)
    parser.add_argument("--rep_seed", type=int, default=0)
    parser.add_argument("--seeds", default="42,1,2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/s3")
    parser.add_argument("--skip_extraction", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="re-run a condition whose study exists"
    )
    parser.add_argument("--transfer", action="store_true")
    args = parser.parse_args()

    wanted = {s for s in args.only.split(",") if s}
    conditions = [c for c in s3_backbones.CONDITIONS if not wanted or c[0] in wanted]
    os.makedirs(args.out, exist_ok=True)
    seeds = args.seeds.replace(",", " ").split()

    if not args.skip_extraction:
        argv = [
            "s3_backbones.py",
            "--device",
            args.device,
            "--out",
            args.out,
            "--latent_dim",
            str(args.latent_dim),
            "--rep_seed",
            str(args.rep_seed),
            "--data_dir",
            args.data_dir,
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.only:
            argv += ["--only", args.only]
        if args.transfer:
            argv += ["--transfer"]
        old = sys.argv
        sys.argv = argv
        try:
            s3_backbones.main()
        finally:
            sys.argv = old

    for name, _arch, _weights in conditions:
        cache = os.path.join(args.out, f"cache_{name}_cifar100", "feature_cache.pt")
        if not os.path.exists(cache):
            print(f"[S3] missing cache for {name}, skipping", flush=True)
            continue
        study_path = os.path.join(args.out, name, f"s2_ladder_study_{name}.json")
        if os.path.exists(study_path) and not args.force:
            # Resume: a server restart during the ~1h ladder phase must not
            # throw away completed conditions.
            print(f"[S3] {name}: study exists, skipping", flush=True)
            continue
        print(f"=== S3 ladder: {name} ===", flush=True)
        t0 = time.time()
        run_with_argv(
            [
                "s2_ladder.py",
                "--cache",
                cache,
                "--levels",
                *LEVELS,
                "--seeds",
                ",".join(seeds),
                "--epochs",
                str(args.epochs),
                "--device",
                args.device,
                "--out",
                os.path.join(args.out, name),
                "--tag",
                f"_{name}",
            ]
        )
        print(f"[S3] {name} finished in {time.time() - t0:.0f}s", flush=True)

    print("S3_DONE", flush=True)


if __name__ == "__main__":
    main()
