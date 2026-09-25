#!/usr/bin/env python3
"""
Single entry point for every Stage 1 run (docs/STAGE1_PLAN.md).

    python experiments/run_all.py --dry-run        # show what would run
    python experiments/run_all.py                  # run everything, skipping done stages
    python experiments/run_all.py --stages s2 s4   # a subset

It is Python rather than bash on purpose: on this machine a bash driver
sourcing the CUDA environment under `set -u` died with no output at all, and a
`pkill -f` pattern matched its own command line. Both failure modes cost hours.
Here the CUDA environment is set once via re-exec (the dynamic loader reads
LD_LIBRARY_PATH at process start, so setting it after `import torch` would be
too late), every stage's output goes to `results/logs/<stage>.log`, and a stage
whose output already exists is skipped unless `--force`.

Stages and their outputs:

    verify   tests, lint, S0 ingest, ViT feature regression      (no output)
    e0       representation ceiling and adapter headroom         results/e0/
    s2       complexity ladder, one backbone, 3 seeds            results/s2/
    s3x      S3 feature extraction (6 backbones x 2 datasets)    results/s3/cache_*
    s3       backbone generalization ladder (needs s3x)          results/s3/*/
    s3r      backbone report, deltas, transfer                   results/s3/s3_backbone_study.json
    s7       representation transfer (needs s3x)                 results/s7/
    s5       protocol axis: Class-IL vs Task-IL, 2x2 factorial    results/s5/
    s5b      domain-incremental rotated MNIST, unseen domain      results/s5b/
    s4       dataset generalization, 4 datasets x 6 levels x 3 seeds  results/s4/
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
LOGS = ROOT / "results" / "logs"
DRIVER_LIB = "/run/opengl-driver/lib"


# ---------------------------------------------------------------------------
# environment: re-exec once with LD_LIBRARY_PATH set
# ---------------------------------------------------------------------------


def ensure_cuda_env() -> None:
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if DRIVER_LIB in current.split(":"):
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{current}:{DRIVER_LIB}" if current else DRIVER_LIB
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:],
        env,
    )


# ---------------------------------------------------------------------------
# stage definitions
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    name: str
    description: str
    commands: list[list[str]]
    done_when: list[Path] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    minutes: float = 0.0

    def is_done(self) -> bool:
        return bool(self.done_when) and all(p.exists() for p in self.done_when)


def _cells_done(path: Path, count: int) -> bool:
    """A study that grows while it runs: existence is not completion."""
    if not path.exists():
        return False
    try:
        with open(path) as fh:
            return len(json.load(fh).get("cells", [])) >= count
    except Exception:
        return False


# Studies whose completion is a cell count rather than the existence of a file.
CELL_TARGETS = {
    "s4": (ROOT / "results" / "s4" / "s4_dataset_study.json", 72),
    "s6": (ROOT / "results" / "s6" / "s6_order_study.json", 45),
    "s6b": (ROOT / "results" / "s6b" / "s6b_difficulty_study.json", 30),
    "s8": (ROOT / "results" / "s8" / "s8_budget_study.json", 56),
    "s9": (ROOT / "results" / "s9" / "s9_robustness_study.json", 170),
    "s10": (ROOT / "results" / "s10" / "s10_scaling_study.json", 160),
}


def stage_is_done(stage: Stage) -> bool:
    target = CELL_TARGETS.get(stage.name)
    if target is not None:
        return _cells_done(*target)
    return stage.is_done()


def build_stages(args) -> list[Stage]:
    seeds = args.seeds
    common = ["--seeds", seeds, "--epochs", str(args.epochs), "--device", args.device]
    s3_conditions = [
        "random",
        "mlp",
        "conv",
        "resnet18_random",
        "resnet18_imagenet",
        "vit_b16_imagenet",
    ]
    return [
        Stage(
            "verify",
            "tests, lint, S0 ingest, ViT feature regression",
            [
                [PY, "-m", "pytest", "tests/", "-q"],
                [PY, "-m", "ruff", "check", "."],
                [PY, "-m", "black", "--check", "."],
                [
                    PY,
                    "-c",
                    "import glob, sys; sys.path.insert(0, '.'); "
                    "from pal_moe.evaluation.schema import load_run_records, validate_run_record; "
                    "ok=bad=0\n"
                    "for f in sorted(glob.glob('results/**/benchmark_results_*.json', recursive=True)):\n"
                    "    for r in load_run_records(f):\n"
                    "        bad += bool(validate_run_record(r)); ok += (not validate_run_record(r))\n"
                    "print(f'{ok} conformant, {bad} rejected')",
                ],
                [PY, "-u", "experiments/s3_check_vit_features.py"],
            ],
            minutes=1,
        ),
        Stage(
            "e0",
            "representation ceiling, adapter headroom, router recall, rejection MVP",
            [
                [
                    PY,
                    "-u",
                    "experiments/e0_representation_ceiling.py",
                    "--mode",
                    "all",
                    "--epochs",
                    str(args.epochs),
                    "--ranks",
                    "0,8,32,64",
                    "--oracle_routing",
                    "--out",
                    "results/e0",
                    "--device",
                    args.device,
                ]
            ],
            done_when=[ROOT / "results" / "e0" / "e0_all_vit_b_16_seed42.json"],
            minutes=30,
        ),
        Stage(
            "s2",
            "complexity ladder on CIFAR-100/ViT, 3 seeds",
            [
                [
                    PY,
                    "-u",
                    "experiments/s2_ladder.py",
                    "--cache",
                    "results/feature_cache/cifar100_vit_b16/feature_cache.pt",
                    *common,
                    "--out",
                    "results/s2",
                ]
            ],
            done_when=[ROOT / "results" / "s2" / "s2_ladder_study.json"],
            minutes=15,
        ),
        Stage(
            "s3x",
            "S3 feature extraction: 6 backbones x (CIFAR-100 + CIFAR-10)",
            [
                [
                    PY,
                    "-u",
                    "experiments/s3_run.py",
                    "--skip_extraction",
                    "--transfer",
                    "--device",
                    args.device,
                ]
            ],
            done_when=[
                ROOT / "results" / "s3" / f"cache_{c}_{d}" / "feature_cache.pt"
                for c in s3_conditions
                for d in ("cifar100", "cifar10")
            ],
            minutes=35,
        ),
        Stage(
            "s3",
            "S3 backbone generalization ladder",
            [
                [
                    PY,
                    "-u",
                    "experiments/s3_run.py",
                    "--transfer",
                    "--seeds",
                    seeds,
                    "--epochs",
                    str(args.epochs),
                    "--device",
                    args.device,
                ]
            ],
            done_when=[
                ROOT / "results" / "s3" / c / f"s2_ladder_study_{c}.json"
                for c in s3_conditions
            ],
            needs=["s3x"],
            minutes=55,
        ),
        Stage(
            "s3r",
            "S3 report: table, deltas, transfer, ViT consistency gate",
            [[PY, "-u", "experiments/s3_report.py"]],
            done_when=[ROOT / "results" / "s3" / "s3_backbone_study.json"],
            needs=["s3"],
            minutes=2,
        ),
        Stage(
            "s7",
            "representation transfer at every checkpoint, raw reference, few-shot",
            [
                [
                    PY,
                    "-u",
                    "experiments/s7_transfer.py",
                    "--backbones",
                    "vit_b16_imagenet,resnet18_imagenet,random",
                    "--levels",
                    "L2b_shared_seq",
                    "L3_per_task",
                    "--seeds",
                    seeds.split(",")[0],
                    "--epochs",
                    str(args.epochs),
                    "--device",
                    args.device,
                    "--out",
                    "results/s7",
                ]
            ],
            done_when=[ROOT / "results" / "s7" / "s7_transfer_cifar10.json"],
            needs=["s3x"],
            minutes=25,
        ),
        Stage(
            "s5",
            "protocol axis: Class-IL / Task-IL 2x2 factorial",
            [
                [
                    PY,
                    "-u",
                    "experiments/s5_protocols.py",
                    "--datasets",
                    "cifar100,cifar10",
                    "--seeds",
                    seeds,
                    "--epochs",
                    str(args.epochs),
                    "--device",
                    args.device,
                    "--out",
                    "results/s5",
                ]
            ],
            done_when=[ROOT / "results" / "s5" / "s5_protocol_study.json"],
            minutes=50,
        ),
        Stage(
            "s5b",
            "Domain-IL: rotated MNIST domains, unseen-domain accuracy",
            [
                [
                    PY,
                    "-u",
                    "experiments/s5b_domains.py",
                    "--seeds",
                    seeds,
                    "--epochs",
                    str(args.epochs),
                    "--device",
                    args.device,
                    "--out",
                    "results/s5b",
                ]
            ],
            done_when=[ROOT / "results" / "s5b" / "s5b_domain_study.json"],
            needs=["s5"],
            minutes=60,
        ),
        Stage(
            "s4",
            "dataset generalization: MNIST / CIFAR-10 / CIFAR-100 / Tiny-ImageNet",
            [
                [
                    PY,
                    "-u",
                    "experiments/s4_datasets.py",
                    "--seeds",
                    seeds,
                    "--epochs",
                    str(args.epochs),
                    "--device",
                    args.device,
                    "--out",
                    "results/s4",
                ]
            ],
            done_when=[],  # handled by the cell count below
            minutes=210,
        ),
        Stage(
            "s6",
            "order sensitivity: class order, task order, unseen task",
            [
                [
                    PY,
                    "-u",
                    "experiments/s6_order.py",
                    "--seeds",
                    seeds.split(",")[0],
                    "--epochs",
                    str(args.epochs),
                    "--device",
                    args.device,
                    "--out",
                    "results/s6",
                ]
            ],
            done_when=[],  # handled by the cell count below
            needs=["s2"],
            minutes=45,
        ),
        Stage(
            "s6b",
            "designed difficulty: coherent vs dispersed task partitions",
            [
                [
                    PY,
                    "-u",
                    "experiments/s6b_difficulty.py",
                    *common,
                    "--out",
                    "results/s6b",
                ]
            ],
            done_when=[],  # handled by the cell count below
            needs=["s6"],
            minutes=45,
        ),
        Stage(
            "s8",
            "resource budget (parameter / memory / active) in both routing regimes",
            [
                [
                    PY,
                    "-u",
                    "experiments/s8_budget.py",
                    "--ranks",
                    "2",
                    "8",
                    "32",
                    "128",
                    "--prototypes",
                    "1",
                    "4",
                    "16",
                    "--topks",
                    "1",
                    "2",
                    "4",
                    *common,
                    "--out",
                    "results/s8",
                ]
            ],
            done_when=[],  # handled by the cell count below
            needs=["s6b"],
            minutes=150,
        ),
        Stage(
            "s8r",
            "S8 report: capability curves, mechanism metrics, resource axes, S6b guard",
            [[PY, "-u", "experiments/s8_report.py"]],
            done_when=[ROOT / "results" / "s8" / "s8_budget_report.json"],
            needs=["s8"],
            minutes=1,
        ),
        Stage(
            "s9x",
            "S9 shift extraction: 9 corruption cells + spurious cue, clean-reproduction guard",
            [
                [
                    PY,
                    "-u",
                    "experiments/s9_corruptions.py",
                    "--extract",
                    "--guard",
                    "--device",
                    args.device,
                ]
            ],
            done_when=[
                ROOT / "results" / "s9" / "cache" / f"{name}_s{severity}.pt"
                for name in ("gaussian_noise", "defocus_blur", "brightness")
                for severity in (1, 3, 5)
            ]
            + [
                ROOT / "results" / "s9" / "cache" / "spurious_train.pt",
                ROOT / "results" / "s9" / "cache" / "spurious_correlated.pt",
                ROOT / "results" / "s9" / "cache" / "spurious_absent.pt",
                ROOT / "results" / "s9" / "cache" / "spurious_flipped.pt",
            ],
            needs=["s8"],
            minutes=40,
        ),
        Stage(
            "s9",
            "robustness: corruption and spurious shift in both routing regimes",
            [
                [
                    PY,
                    "-u",
                    "experiments/s9_robustness.py",
                    *common,
                    "--out",
                    "results/s9",
                ]
            ],
            done_when=[],  # handled by the cell count below
            needs=["s9x"],
            minutes=40,
        ),
        Stage(
            "s10",
            "scalability: task/expert count, two datasets, candidate-set control",
            [
                [
                    PY,
                    "-u",
                    "experiments/s10_scaling.py",
                    "--datasets",
                    "cifar100",
                    "tinyimagenet",
                    "--constructs",
                    "contiguous",
                    "dispersed",
                    "--seeds",
                    seeds,
                    "--epochs",
                    str(args.epochs),
                    "--device",
                    args.device,
                    "--out",
                    "results/s10",
                ]
            ],
            done_when=[],  # handled by the cell count below
            needs=["s9"],
            minutes=60,
        ),
    ]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_stage(stage: Stage, force: bool, dry: bool) -> dict:
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{stage.name}.log"

    done = stage_is_done(stage)
    if done and not force:
        print(f"  SKIP  {stage.name:6s} (output exists)")
        return {
            "stage": stage.name,
            "status": "skipped",
            "seconds": 0.0,
            "log": str(log_path),
        }

    print(f"  RUN   {stage.name:6s} ~{stage.minutes:.0f} min  {stage.description}")
    if dry:
        for cmd in stage.commands:
            print("        " + " ".join(cmd))
        return {
            "stage": stage.name,
            "status": "dry-run",
            "seconds": 0.0,
            "log": str(log_path),
        }

    started = time.time()
    with open(log_path, "a") as log:
        log.write(
            f"\n===== {datetime.datetime.now().isoformat(timespec='seconds')} {stage.name} =====\n"
        )
        log.flush()
        for cmd in stage.commands:
            print(
                f"        $ {' '.join(cmd[:6])}{' ...' if len(cmd) > 6 else ''}",
                flush=True,
            )
            completed = subprocess.run(
                cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            if completed.returncode != 0:
                print(
                    f"  FAIL  {stage.name:6s} (exit {completed.returncode})  see {log_path}"
                )
                return {
                    "stage": stage.name,
                    "status": f"failed({completed.returncode})",
                    "seconds": round(time.time() - started, 1),
                    "log": str(log_path),
                }
    elapsed = round(time.time() - started, 1)
    print(f"  DONE  {stage.name:6s} {elapsed / 60:.1f} min")
    return {
        "stage": stage.name,
        "status": "done",
        "seconds": elapsed,
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stages", nargs="*", default=["all"], help="stage names, or 'all'"
    )
    parser.add_argument("--seeds", default="42,1,2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--force", action="store_true", help="re-run stages whose output exists"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands, run nothing"
    )
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    args = parser.parse_args()

    ensure_cuda_env()

    stages = build_stages(args)
    by_name = {s.name: s for s in stages}
    if args.list:
        for stage in stages:
            mark = "done" if stage_is_done(stage) else "todo"
            print(
                f"  {stage.name:6s} [{mark}] ~{stage.minutes:5.0f} min  {stage.description}"
            )
        return

    wanted = args.stages
    if "all" in wanted:
        selected = [s.name for s in stages]
    else:
        unknown = [n for n in wanted if n not in by_name]
        if unknown:
            raise SystemExit(f"unknown stages {unknown}; available: {list(by_name)}")
        selected = []
        for stage in stages:  # keep the declared order, pull in dependencies
            if stage.name in wanted or any(d in selected for d in stage.needs):
                selected.append(stage.name)

    print(
        f"\nStage 1 runner: {len(selected)} stages, device={args.device}, seeds={args.seeds}"
    )
    print("=" * 78)
    results = []
    for name in selected:
        stage = by_name[name]
        missing = [
            d for d in stage.needs if d not in selected and not by_name[d].is_done()
        ]
        if missing:
            print(f"  WARN  {name} needs {missing} which are neither selected nor done")
        results.append(run_stage(stage, force=args.force, dry=args.dry_run))

    print("=" * 78)
    print(f"{'stage':8s} {'status':12s} {'minutes':>8s}  log")
    for row in results:
        print(
            f"{row['stage']:8s} {row['status']:12s} {row['seconds'] / 60:8.1f}  {row['log']}"
        )
    print()
    total = sum(r["seconds"] for r in results)
    print(f"total {total / 60:.1f} min")
    if not args.dry_run:
        print("\nNext: read the studies under results/<stage>/ and docs/STAGE1_PLAN.md")


if __name__ == "__main__":
    main()
