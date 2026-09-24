"""
Regression check: the S1 backbone factory must reproduce the runner-built ViT
features.

The bug this guards against (found in S3, docs/STAGE1_PLAN.md 5.3): asking a
ViT-B/16 encoder for `output_dim=256` inserts a 768->256 BatchNorm+ReLU head and
then projects back up, which collapses the CIFAR-100 features (norm 4.3 instead
of 18.4, NCM 28% instead of 70%) without anything in the accuracy table looking
wrong. The factory now asks for the backbone's **native** width, so the ViT
needs no projection and its features must match the reference cache.

Run as a file (not stdin) so DataLoader workers can spawn.

Usage:
    python experiments/s3_check_vit_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from pal_moe.arch import build_projected_backbone  # noqa: E402
from pal_moe.data.split_cifar100 import get_split_cifar100_tasks  # noqa: E402

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
REFERENCE_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"
# Row order differs between cache builds (the loader shuffles), so compare the
# distribution, not the tensors.
TOLERANCE = 0.5


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = (
        build_projected_backbone(
            "vit_b_16",
            input_dim=3072,
            latent_dim=768,
            backbone_weights="imagenet",
            input_mean=CIFAR100_MEAN,
            input_std=CIFAR100_STD,
        )
        .to(device)
        .eval()
    )
    inner = backbone.encoder if hasattr(backbone, "encoder") else backbone
    identity_head = isinstance(inner.net[-1], torch.nn.Identity)
    tasks = get_split_cifar100_tasks(
        data_dir="./data", batch_size=128, val_split=0.1, seed=42, num_workers=0
    )
    with torch.no_grad():
        z = torch.cat(
            [
                backbone.encode(x.to(device)).cpu()
                for i, (x, _y) in enumerate(tasks[0].train_loader)
                if i < 4
            ]
        ).float()
    reference = torch.load(REFERENCE_CACHE, map_location="cpu", weights_only=True)[
        "tasks"
    ][0]["splits"]["train"][0].float()

    ours, ref = float(z.norm(dim=1).mean()), float(reference.norm(dim=1).mean())
    print(f"identity head: {identity_head}")
    print(f"factory norm : {ours:6.2f}")
    print(f"reference    : {ref:6.2f}")
    print(f"tolerance    : {TOLERANCE}")
    ok = identity_head and abs(ours - ref) < TOLERANCE
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
