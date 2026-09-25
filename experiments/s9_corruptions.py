"""
S9 shifts: controlled pixel-level corruptions and a spurious cue, extracted
through the *same* pipeline as the canonical cache.

The whole point of S9 is whether the S5b/S6b/S8 decomposition survives a
distribution shift, so the shift must be the only variable. Two families:

    corruption   test-time only, clean training: gaussian noise, defocus blur,
                 brightness, each at three severities (a controlled CIFAR-C
                 style family, not the official CIFAR-100-C benchmark)
    spurious     a 6x6 red cue in the image corner, present during training on
                 the first half of the tasks; at test time it is correlated
                 (as trained), absent, or flipped onto the other half

A `ShiftedCIFAR100` applies the shift to the PIL image *before* the transform
chain, so the tail of the pipeline (`ToTensor` + `Normalize`) is byte-identical
to the clean path, and `get_split_cifar100_tasks` keeps the same partition,
index draw and class order - a shifted cache is paired sample-for-sample with
the clean one.

Usage (from `experiments/s9_robustness.py`):
    python experiments/s9_corruptions.py --extract --device cuda
"""

import argparse
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import datasets  # noqa: E402

from pal_moe.arch import build_projected_backbone  # noqa: E402
from pal_moe.data.feature_cache import _encode_split  # noqa: E402
from pal_moe.data.split_cifar100 import get_split_cifar100_tasks  # noqa: E402

CANONICAL_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"
SHIFT_ROOT = "results/s9/cache"
DATASET_MEAN = (0.5071, 0.4867, 0.4408)
DATASET_STD = (0.2675, 0.2565, 0.2761)

# severity 1 / 3 / 5 of a controlled CIFAR-C style family. The parameters are
# documented rather than tuned: the point is a monotone severity axis, not a
# reproduction of the official benchmark.
CORRUPTIONS = {
    "gaussian_noise": {"kind": "noise", "params": [0.05, 0.10, 0.18]},
    "defocus_blur": {"kind": "blur", "params": [1, 2, 3]},
    "brightness": {"kind": "gain", "params": [1.3, 1.6, 2.0]},
}
SEVERITIES = [1, 3, 5]
CUE_SIZE = 6
CUE_TASK_GROUP = 10  # tasks 0..9 = classes 0..49 carry the cue during training
SPURIOUS_MODES = ["correlated", "absent", "flipped"]


# ---------------------------------------------------------------------------
# the shift, applied to the PIL image before ToTensor/Normalize
# ---------------------------------------------------------------------------


class ShiftedCIFAR100(datasets.CIFAR100):
    """CIFAR-100 whose images pass through `shift(img, target, index)` first.

    Mirrors `torchvision.datasets.CIFAR100.__getitem__` exactly, with the shift
    inserted between the PIL image and the transform chain, so everything after
    the shift is the canonical preprocessing.

    The shift is an *instance* attribute and the shift objects are module-level
    dataclasses, because a DataLoader worker under Python 3.14's forkserver
    re-imports the module: a class attribute or a dynamically created class
    would not survive the trip to the worker.
    """

    def __init__(self, *args, shift=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift = shift

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.shift is not None:
            img = self.shift(img, int(target), int(index))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


@dataclass
class CorruptionShift:
    """One (corruption, severity) cell, deterministic per image index."""

    kind: str
    param: float

    def __call__(self, img: Image.Image, target: int, index: int) -> Image.Image:
        x = _to_tensor(img)
        return _to_pil(_apply_corruption(x, self.kind, self.param, index))


@dataclass
class SpuriousShift:
    """The corner cue, present according to `mode`."""

    mode: str

    def __call__(self, img: Image.Image, target: int, index: int) -> Image.Image:
        in_group = target < CUE_TASK_GROUP * 5
        present = {
            "correlated": in_group,
            "absent": False,
            "flipped": not in_group,
        }[self.mode]
        if not present:
            return img
        x = _to_tensor(img)
        x[:, 0, :CUE_SIZE, :CUE_SIZE] = 1.0
        x[:, 1, :CUE_SIZE, :CUE_SIZE] = 0.0
        x[:, 2, :CUE_SIZE, :CUE_SIZE] = 0.0
        return _to_pil(x)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(img))
        .permute(2, 0, 1)
        .float()
        .div_(255)
        .unsqueeze(0)
    )


def _to_pil(x: torch.Tensor) -> Image.Image:
    array = x.squeeze(0).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def _disk_kernel(radius: int) -> torch.Tensor:
    size = 2 * radius + 1
    axis = torch.arange(size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    mask = ((yy - radius) ** 2 + (xx - radius) ** 2) <= radius**2
    kernel = mask.float()
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size).expand(3, 1, size, size)


def _apply_corruption(
    x: torch.Tensor, kind: str, param: float, index: int
) -> torch.Tensor:
    if kind == "noise":
        generator = torch.Generator().manual_seed(index)
        return x + param * torch.randn(x.shape, generator=generator)
    if kind == "blur":
        return F.conv2d(x, _disk_kernel(int(param)), padding=int(param), groups=3)
    if kind == "gain":
        return x * param
    raise KeyError(f"unknown corruption kind {kind!r}")


def corruption_shift(name: str, severity: int) -> CorruptionShift:
    """A shift object for one (corruption, severity) cell."""
    spec = CORRUPTIONS[name]
    param = spec["params"][SEVERITIES.index(severity)]
    return CorruptionShift(kind=spec["kind"], param=float(param))


def spurious_shift(mode: str) -> SpuriousShift:
    """The cue, present according to `mode` (`correlated`/`absent`/`flipped`)."""
    if mode not in SPURIOUS_MODES:
        raise KeyError(f"unknown spurious mode {mode!r}")
    return SpuriousShift(mode=mode)


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def build_backbone(device, latent_dim: int = 768):
    backbone = build_projected_backbone(
        "vit_b_16",
        input_dim=3 * 32 * 32,
        latent_dim=latent_dim,
        rep_seed=0,
        backbone_weights="imagenet",
        input_mean=DATASET_MEAN,
        input_std=DATASET_STD,
    ).to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


def _dataset_cls(shift):
    """The shifted dataset class, carrying one picklable shift object."""
    if shift is None:
        return ShiftedCIFAR100
    return partial(ShiftedCIFAR100, shift=shift)


def shifted_tasks(
    shift_test=None,
    shift_train=None,
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
):
    """The canonical split with shifts injected, index-for-index identical.

    `get_split_cifar100_tasks` seeds the RNG itself, so two calls with the same
    seed draw the same partition: the train and test shifts are therefore
    extracted from two calls and the loaders are grafted, which keeps the
    splitter as the single source of truth for the partition.
    """
    base = dict(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
    tasks_train = get_split_cifar100_tasks(
        dataset_cls=_dataset_cls(shift_train), **base
    )
    if shift_test is None:
        return tasks_train
    tasks_test = get_split_cifar100_tasks(dataset_cls=_dataset_cls(shift_test), **base)
    grafted = []
    for train_task, test_task in zip(tasks_train, tasks_test):
        assert tuple(train_task.classes) == tuple(test_task.classes)
        train_task.test_loader = test_task.test_loader
        grafted.append(train_task)
    return grafted


def extract_test(backbone, tasks, device, dtype=torch.float16) -> dict:
    """Per-task test features under whatever shift the task loaders carry."""
    out = {}
    for task in tasks:
        feats, labels = _encode_split(backbone, task.test_loader, device, dtype)
        out[int(task.task_id)] = (feats, labels)
    return out


def extract_all(backbone, tasks, device, dtype=torch.float16) -> dict:
    out = {}
    for task in tasks:
        row = {}
        for split, loader in (
            ("train", task.train_loader),
            ("val", task.val_loader),
            ("test", task.test_loader),
        ):
            feats, labels = _encode_split(backbone, loader, device, dtype)
            row[split] = (feats, labels)
        out[int(task.task_id)] = row
    return out


def save_shift_cache(path: str, meta: dict, per_task: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"meta": dict(meta), "tasks": []}
    for task_id in sorted(per_task):
        row = {"task_id": task_id, "splits": {}}
        for split, value in per_task[task_id].items():
            row["splits"][split] = value
        payload["tasks"].append(row)
    torch.save(payload, path)


def load_shift_cache(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "meta": payload["meta"],
        "tasks": {
            int(row["task_id"]): {
                name: (value[0].float(), value[1].long())
                for name, value in row["splits"].items()
            }
            for row in payload["tasks"]
        },
    }


def cache_path(name: str, severity=None) -> str:
    if severity is None:
        return os.path.join(SHIFT_ROOT, f"{name}.pt")
    return os.path.join(SHIFT_ROOT, f"{name}_s{severity}.pt")


def condition_shifts() -> dict[str, object]:
    """Every S9 test condition and the shift that defines it."""
    shifts = {}
    for name in CORRUPTIONS:
        for severity in SEVERITIES:
            shifts[f"{name}_s{severity}"] = corruption_shift(name, severity)
    for mode in SPURIOUS_MODES:
        shifts[f"spurious_{mode}"] = spurious_shift(mode)
    return shifts


def clean_guard(device, batch_size: int = 128, num_workers: int = 0) -> dict:
    """Re-extract the clean test split and compare with the canonical cache.

    This is the guard that makes the shifted caches trustworthy: if the clean
    pass does not reproduce the canonical test features, the pipeline has
    drifted and every shifted number would be confounded.
    """
    backbone = build_backbone(device)
    tasks = shifted_tasks(batch_size=batch_size, num_workers=num_workers)
    here = extract_test(backbone, tasks, device)
    canonical = torch.load(CANONICAL_CACHE, map_location="cpu", weights_only=True)
    worst = 0.0
    checked = 0
    for row in canonical["tasks"]:
        task_id = int(row["task_id"])
        ref = row["splits"]["test"][0].float()
        got = here[task_id][0].float()
        if ref.shape != got.shape:
            return {"matched": False, "reason": f"shape {ref.shape} vs {got.shape}"}
        worst = max(worst, float((ref - got).abs().max()))
        checked += ref.size(0)
    return {"matched": worst < 1e-3, "max_abs_delta": worst, "samples": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=768)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--guard", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.guard:
        report = clean_guard(device, args.batch_size, args.num_workers)
        print(f"[S9 guard] clean re-extraction vs canonical: {report}")
        if not report.get("matched"):
            raise SystemExit("clean pass does not reproduce the canonical cache")
    if not args.extract:
        return

    backbone = build_backbone(device, args.latent_dim)
    base = dict(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # corruption family: clean training, shifted test only
    for name in CORRUPTIONS:
        for severity in SEVERITIES:
            path = cache_path(name, severity)
            if os.path.exists(path) and not args.force:
                print(f"[S9] {name}_s{severity}: exists, skipping")
                continue
            tasks = shifted_tasks(shift_test=corruption_shift(name, severity), **base)
            per_task = {
                task_id: {"test": value}
                for task_id, value in extract_test(backbone, tasks, device).items()
            }
            save_shift_cache(
                path,
                {
                    "family": "corruption",
                    "name": name,
                    "severity": severity,
                    "applied_to": "test",
                    "training": "clean",
                },
                per_task,
            )
            print(f"[S9] {name}_s{severity}: wrote {path}")

    # spurious family: the cue is in training, so the train split is extracted
    path = cache_path("spurious_train")
    if os.path.exists(path) and not args.force:
        print("[S9] spurious_train: exists, skipping")
    else:
        tasks = shifted_tasks(shift_train=spurious_shift("correlated"), **base)
        save_shift_cache(
            path,
            {"family": "spurious", "applied_to": "train", "mode": "correlated"},
            extract_all(backbone, tasks, device),
        )
        print(f"[S9] spurious_train: wrote {path}")

    for mode in SPURIOUS_MODES:
        path = cache_path(f"spurious_{mode}")
        if os.path.exists(path) and not args.force:
            print(f"[S9] spurious_{mode}: exists, skipping")
            continue
        tasks = shifted_tasks(shift_test=spurious_shift(mode), **base)
        per_task = {
            task_id: {"test": value}
            for task_id, value in extract_test(backbone, tasks, device).items()
        }
        save_shift_cache(
            path,
            {"family": "spurious", "applied_to": "test", "mode": mode},
            per_task,
        )
        print(f"[S9] spurious_{mode}: wrote {path}")


if __name__ == "__main__":
    main()
