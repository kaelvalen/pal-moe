"""
Flatten the Tiny-ImageNet train split into an ImageFolder-compatible tree.

The official archive stores each class as::

    train/n01443537/images/n01443537_0.JPEG
    train/n01443537/n01443537_boxes.txt

``torchvision.datasets.ImageFolder`` needs the images directly inside the class
folder, so this script creates::

    train_flat/n01443537/n01443537_0.JPEG

Symlinks by default (no extra disk), copies with ``--copy``. Class order is
alphabetical, which matches the ImageFolder ordering used by the runner.

Usage:
    python experiments/prepare_tiny_imagenet.py \
        --source data/tiny-imagenet-200/train \
        --dest data/tiny-imagenet-200/train_flat
"""

import argparse
import os
import shutil
from pathlib import Path

IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png", ".bmp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/tiny-imagenet-200/train")
    parser.add_argument("--dest", type=str, default="data/tiny-imagenet-200/train_flat")
    parser.add_argument("--copy", action="store_true", help="copy files instead of symlinks")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    if not source.is_dir():
        raise SystemExit(f"source directory not found: {source}")
    dest.mkdir(parents=True, exist_ok=True)

    created = 0
    for class_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        images = class_dir / "images"
        if not images.is_dir():
            continue
        out_dir = dest / class_dir.name
        out_dir.mkdir(exist_ok=True)
        for image in sorted(images.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            target = out_dir / image.name
            if target.exists():
                continue
            if args.copy:
                shutil.copy2(image, target)
            else:
                os.symlink(image.resolve(), target)
            created += 1

    print(f"flattened {created} images from {source} into {dest}")


if __name__ == "__main__":
    main()
