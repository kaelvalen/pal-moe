"""
S3 - backbone generalization under a fixed latent dimension.

The S2 result made this experiment necessary:

    L1_ridge (0 params)  76.77%  >  L3_per_task (89K)  70.58%
    L2b -> L3            +14.58   (isolation helps)
    L4 - L3              26.88    (routing tax)

So the question is no longer "which backbone is best" but: does that ordering
survive a change of representation family? Concretely, S3 answers four questions
(docs/STAGE1_PLAN.md section 5):

    S3.1 readout dominance   is `ridge > MoE` true on every backbone?
    S3.2 representation dependence  how does `L3 - ridge` move with the backbone?
    S3.3 routing dependence  how does the routing tax `L4 - L3` move?
    S3.4 oracle capacity     is there capacity at all (`L4 >> L3`) or not?

Two controls that S2 did not need:

- **fixed latent dimension.** Different backbones have different native widths
  (MLP 256, ResNet 512, ViT 768, pixels 3072). Sweeping the backbone without a
  common `d0` would change the readout's parameter count, the prototype store
  and the effective capacity at the same time. Every backbone is therefore
  followed by a frozen, seeded projection to `--latent_dim` (no trainable model
  enters).
- **representation seed != training seed.** For a random backbone, `--rep_seed`
  controls the representation and `--seed` controls the CL run. Conflating them
  makes the results uninterpretable.

This script only builds the caches; the ladder itself is run by
`experiments/s2_ladder.py` on each cache, so S2 and S3 use byte-identical
ladder code.

Usage:
    python experiments/s3_backbones.py --device cuda
    python experiments/s3_backbones.py --only random,mlp --device cuda
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from pal_moe.arch import build_projected_backbone, describe
from pal_moe.data.feature_cache import build_feature_cache, save_feature_cache
from pal_moe.data.split_cifar import get_split_cifar10_tasks
from pal_moe.data.split_cifar100 import get_split_cifar100_tasks

# The normalisation each splitter applies. The ViT path undoes it, so a
# backbone factory that does not forward it double-normalises (S3 measured the
# collapse: L3 28.9% instead of 70.6%).
DATASET_NORM = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

# (name, arch, backbone_weights) - the "random + existing pretrained family"
# sweep the brief asks for. Architecture x pretraining as a full factorial is a
# later experiment; ResNet-18 carries the contrast here.
CONDITIONS = [
    ("random", "random", "none"),
    ("mlp", "mlp", "none"),
    ("conv", "conv", "none"),
    ("resnet18_random", "resnet18", "none"),
    ("resnet18_imagenet", "resnet18", "imagenet"),
    ("vit_b16_imagenet", "vit_b_16", "imagenet"),
]


def build_tasks(dataset: str, args, seed: int):
    common = dict(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=seed,
        num_workers=args.num_workers,
    )
    if dataset == "cifar100":
        return get_split_cifar100_tasks(**common)
    return get_split_cifar10_tasks(**common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma-separated condition names")
    parser.add_argument("--latent_dim", type=int, default=768)
    parser.add_argument("--rep_seed", type=int, default=0)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/s3")
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="also extract a CIFAR-10 cache for the representation-transfer axis",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    wanted = {s for s in args.only.split(",") if s}
    conditions = [c for c in CONDITIONS if not wanted or c[0] in wanted]
    os.makedirs(args.out, exist_ok=True)
    print(f"[S3] latent_dim={args.latent_dim} rep_seed={args.rep_seed}")
    print(f"[S3] registry: {json.dumps(describe())}")

    index = []
    for name, arch, weights in conditions:
        for dataset, split in (("cifar100", "cifar100"),) + (
            (("cifar10", "cifar10"),) if args.transfer else ()
        ):
            cache_dir = os.path.join(args.out, f"cache_{name}_{dataset}")
            cache_path = os.path.join(cache_dir, "feature_cache.pt")
            if os.path.exists(cache_path):
                print(f"[S3] {name}/{dataset}: cache exists, skipping extraction")
                index.append(
                    {
                        "condition": name,
                        "arch": arch,
                        "weights": weights,
                        "dataset": dataset,
                        "cache": cache_path,
                        "reused": True,
                    }
                )
                continue
            t0 = time.time()
            # The dataset statistics are mandatory for the ViT path, which
            # undoes them before applying ImageNet normalisation.
            mean, std = DATASET_NORM[dataset]
            backbone = build_projected_backbone(
                arch,
                input_dim=3 * 32 * 32,
                latent_dim=args.latent_dim,
                rep_seed=args.rep_seed,
                backbone_weights=weights,
                input_mean=mean,
                input_std=std,
            ).to(device)
            backbone.eval()
            for param in backbone.parameters():
                param.requires_grad = False
            tasks = build_tasks(split, args, seed=42)
            cache = build_feature_cache(
                backbone,
                tasks,
                device,
                num_workers=args.num_workers,
                verbose=False,
            )
            save_feature_cache(
                cache,
                cache_path,
                meta={
                    "dataset": dataset,
                    "encoder_arch": arch,
                    "encoder_weights": weights,
                    "feature_dim": int(cache.feature_dim),
                    "latent_dim": args.latent_dim,
                    "rep_seed": args.rep_seed,
                    "num_tasks": len(tasks),
                    "projected": True,
                    "seed": None,
                },
            )
            print(
                f"[S3] {name}/{dataset}: extracted {len(tasks)} tasks, "
                f"dim={cache.feature_dim}, {time.time() - t0:.1f}s"
            )
            index.append(
                {
                    "condition": name,
                    "arch": arch,
                    "weights": weights,
                    "dataset": dataset,
                    "cache": cache_path,
                    "reused": False,
                }
            )

    path = os.path.join(args.out, "s3_caches.json")
    with open(path, "w") as fh:
        json.dump(
            {
                "latent_dim": args.latent_dim,
                "rep_seed": args.rep_seed,
                "conditions": index,
            },
            fh,
            indent=1,
        )
    print(f"[S3] wrote {path}")


if __name__ == "__main__":
    main()
