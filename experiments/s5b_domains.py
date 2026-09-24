"""
S5b - Domain-IL: does expert isolation help across *distribution shifts*, or was
the benefit in S2-S5 mainly the class-space problem of Class-IL?

Domain-IL keeps the label space fixed and changes the input distribution, so
`class_space` stays `shared` and class masking is a no-op **by construction**.
That removes the explanation S5 pointed at (P2/P3: the class-space restriction
and the routing decision carry the same information) and leaves the question
that matters:

    is isolation useful between shifted distributions?

Design:

- Domains are rotations of MNIST (0/90/180/270 in the stream, 45 held out), the
  standard domain-incremental stress test: the classes keep their identity while
  the input distribution moves.
- Each domain carries **all 10 classes**, which is what makes this Domain-IL
  rather than a relabelled Class-IL stream.
- Feature caches are built with the canonical frozen ViT-B/16 so the S2-S5
  chain stays comparable; the held-out domain is a fifth cache that the
  continual run never sees.
- Each (level, seed) trains once and is evaluated under both routing modes,
  because training is protocol-independent, and the three-way comparison the
  stage is for is `L1_ridge` / `L2b` / `L3`.

Reported per cell: accuracy, forgetting, BWT, FWT, the routing effect,
`MI(Expert;Domain)` and `MI(Expert;Class)`, expert reuse, and accuracy on the
**unseen domain**.

Usage:
    python experiments/s5b_domains.py --device cuda
    python experiments/s5b_domains.py --levels L1_ridge L2b_shared_seq L3_per_task --seeds 42
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import s2_ladder  # noqa: E402
import s5_protocols  # noqa: E402
from pal_moe.arch import build_projected_backbone  # noqa: E402
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402

LEVELS = [
    "L0_ncm",
    "L1_ridge",
    "L2a_shared_joint",
    "L2b_shared_seq",
    "L3_per_task",
    "L4_oracle",
]
STREAM_DOMAINS = [0, 90, 180, 270]
UNSEEN_DOMAIN = 45
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
TRAIN_PER_DOMAIN = 15000


# ---------------------------------------------------------------------------
# domain caches
# ---------------------------------------------------------------------------


def build_domain_cache(degrees: int, path: str, args, device, verbose: bool = True):
    """One cache = one task = one domain, containing all 10 classes."""
    from torchvision import datasets, transforms

    if os.path.exists(path):
        if verbose:
            print(f"[S5b] domain {degrees:3d}: cache exists", flush=True)
        return path
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    )
    train = datasets.MNIST(
        args.data_dir, train=True, download=False, transform=transform
    )
    test = datasets.MNIST(
        args.data_dir, train=False, download=False, transform=transform
    )
    generator = torch.Generator().manual_seed(0)

    def rotated(feats: torch.Tensor) -> torch.Tensor:
        """90-degree multiples are exact pixel rotations; other angles are
        interpolated, so the held-out domain is a genuinely different
        distribution rather than a relabelled copy of one in the stream.

        The original version used `k = degrees // 90`, so a 45-degree
        "held-out domain" silently became `k = 0` - the unrotated domain, which
        is already in the stream, making every unseen-domain number equal to a
        stream number.
        """
        if degrees % 90 == 0:
            if degrees % 360 == 0:
                return feats
            return torch.rot90(feats, k=(degrees // 90) % 4, dims=[-2, -1])
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        return TF.rotate(
            feats, float(degrees), interpolation=InterpolationMode.BILINEAR
        )

    def stack(dataset, limit=None):
        xs, ys = [], []
        for index in range(len(dataset)):
            x, y = dataset[index]
            xs.append(x)
            ys.append(y)
            if limit is not None and len(xs) >= limit:
                break
        return torch.stack(xs), torch.tensor(ys)

    # Class-balanced subsample of the training split; the full test split is
    # kept because the accuracy is the measurement.
    xs, ys = stack(train)
    per_class = max(1, TRAIN_PER_DOMAIN // 10)
    picked = []
    for c in range(10):
        pool = (ys == c).nonzero(as_tuple=True)[0]
        picked.append(
            pool[torch.randperm(pool.numel(), generator=generator)[:per_class]]
        )
    picked = torch.cat(picked)
    # The rotation must be applied to BOTH splits. Applying it only to train
    # (the original bug) left every domain with the same unrotated test set, so
    # "domain accuracy" and "unseen domain accuracy" were the same measurement
    # and the whole stage was meaningless.
    train_x, train_y = rotated(xs[picked]), ys[picked]
    test_x_raw, test_y = stack(test)
    test_x = rotated(test_x_raw)
    if degrees % 360 != 0:
        assert not torch.equal(
            test_x, test_x_raw
        ), f"domain {degrees}: rotation did not change the test split"

    features = {}
    backbone = (
        build_projected_backbone(
            "vit_b_16",
            input_dim=3 * 32 * 32,
            latent_dim=768,
            backbone_weights="imagenet",
            input_mean=MNIST_MEAN,
            input_std=MNIST_STD,
        )
        .to(device)
        .eval()
    )
    with torch.no_grad():
        for name, x in (("train", train_x), ("test", test_x)):
            chunks = []
            for start in range(0, x.size(0), 128):
                chunks.append(
                    backbone.encode(x[start : start + 128].to(device)).cpu().half()
                )
            features[name] = torch.cat(chunks)

    payload = {
        "meta": {
            "dataset": "mnist_rotated",
            "encoder_arch": "vit_b_16",
            "encoder_weights": "imagenet",
            "feature_dim": 768,
            "latent_dim": 768,
            "rep_seed": 0,
            "num_tasks": 1,
            "domain_degrees": degrees,
            "train_per_domain": int(train_x.size(0)),
            "projected": True,
            "seed": None,
        },
        "tasks": [
            {
                "task_id": 0,
                "classes": list(range(10)),
                "splits": {
                    # The ladder uses `val` only for nothing; reuse test to keep
                    # the cache small and the splits honest.
                    "train": (features["train"], train_y),
                    "val": (features["test"], test_y),
                    "test": (features["test"], test_y),
                },
            }
        ],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    if verbose:
        print(
            f"[S5b] domain {degrees:3d}: cache built ({train_x.size(0)} train rows)",
            flush=True,
        )
    return path


def combine_stream(paths: list[str], out_path: str) -> str:
    """Merge per-domain caches into one multi-task cache (a task per domain)."""
    if os.path.exists(out_path):
        return out_path
    tasks = []
    meta = None
    for index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        meta = meta or payload["meta"]
        row = payload["tasks"][0]
        tasks.append(
            {
                "task_id": index,
                "classes": row["classes"],
                "splits": row["splits"],
            }
        )
    payload = {
        "meta": {**meta, "num_tasks": len(tasks), "stream": True},
        "tasks": tasks,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)
    return out_path


# ---------------------------------------------------------------------------
# one cell: train once, evaluate under both routing modes
# ---------------------------------------------------------------------------


def run_cell(
    level: str, stream_cache: str, unseen_cache: str, args, device
) -> list[dict]:
    meta, tasks = s2_ladder.load_tasks(stream_cache)
    unseen_meta, unseen_tasks = s2_ladder.load_tasks(unseen_cache)
    dim = int(meta["feature_dim"])
    num_classes = sum(len(t["classes"]) for t in tasks)
    spec = s2_ladder.LEVELS_BY_NAME[level]

    s2_ladder.set_seed(args.seed)
    per_task = len(tasks[0]["classes"])
    model = s2_ladder.LadderModel(
        spec,
        dim,
        num_classes,
        args,
        device,
        router_slots=len(tasks) * per_task,
    )
    n = len(tasks)
    fwt: list[float] = []
    if spec.joint:
        all_feats = torch.cat([t["splits"]["train"][0] for t in tasks], dim=0)
        all_labels = torch.cat([t["splits"]["train"][1] for t in tasks], dim=0)
        model.seen = sorted({c for t in tasks for c in t["classes"]})
        for i, task in enumerate(tasks):
            model.register_task(task, i, router_offset=i * per_task)
        model.fit_task(None, 0, joint_data=(all_feats, all_labels))
        if spec.readout in s2_ladder.CLOSED_FORM_READOUTS:
            model.readout.fit(all_feats, all_labels, seen_classes=model.seen)
    else:
        for t, task in enumerate(tasks):
            if model.seen:
                _a, delta = s2_ladder.forward_transfer(model, task, num_classes, dim)
                fwt.append(delta)
            model.seen = sorted(set(model.seen) | set(task["classes"]))
            model.fit_task(task, t)
            model.register_task(task, t, router_offset=t * per_task)

    mi = s5_protocols.expert_entropy_mi(model, tasks, device)

    cells = []
    for protocol, oracle in (("domain_il", False), ("domain_il_oracle", True)):
        R = np.zeros((n, n), dtype=np.float32)
        if spec.joint:
            for i in range(n):
                R[n - 1, i] = model.evaluate_task(tasks[i], oracle=oracle)
        else:
            for t in range(n):
                for i in range(t + 1):
                    R[t, i] = model.evaluate_task(tasks[i], oracle=oracle, task_id=i)
        T = n - 1
        forget = [
            max(0.0, float(np.max(R[i : T + 1, i])) - float(R[T, i])) for i in range(T)
        ]
        unseen = float(
            np.mean([model.evaluate_task(t, oracle=oracle) for t in unseen_tasks])
        )
        # Expert reuse: how many distinct experts the router actually selects.
        reuse = None
        if model.router is not None and model.experts:
            used = set()
            with torch.no_grad():
                for task in tasks:
                    ids, _ = model.router.top_k(
                        task["splits"]["test"][0].to(device), k=1
                    )
                    used.update(ids.flatten().tolist())
            reuse = len(used) / max(len(model.experts), 1)
        cells.append(
            {
                "level": level,
                "seed": args.seed,
                "protocol": protocol,
                "routing_mode": "oracle" if oracle else "learned",
                "task_id_at_inference": bool(oracle),
                "class_masking": False,
                "increment_type": "domain",
                "class_space": "shared",
                "num_tasks": n,
                "classes_per_task": len(tasks[0]["classes"]),
                "accuracy": float(np.mean(R[T, :])),
                "forgetting": float(np.mean(forget)) if forget else 0.0,
                "bwt": (
                    float(np.mean([R[T, i] - R[i, i] for i in range(T)])) if T else 0.0
                ),
                "fwt": float(np.mean(fwt)) if fwt else None,
                "acc_matrix": R.tolist(),
                "unseen_domain_accuracy": unseen,
                "expert_reuse": reuse,
                "params": model.cost()["params"],
                "expert_count": 0 if not spec.expert else n,
                **mi,
            }
        )
    return cells


def aggregate(cells: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = {}
    for cell in cells:
        grouped.setdefault((cell["level"], cell["protocol"]), []).append(cell)

    def st(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        if len(values) == 1:
            return {"mean": float(values[0]), "std": 0.0, "n": 1}
        return {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.stdev(values)),
            "n": len(values),
        }

    levels: dict[str, dict] = {}
    for (level, protocol), rows in grouped.items():
        levels.setdefault(level, {})[protocol] = {
            "accuracy": st([r["accuracy"] for r in rows]),
            "forgetting": st([r["forgetting"] for r in rows]),
            "unseen_domain": st([r["unseen_domain_accuracy"] for r in rows]),
            "expert_reuse": st([r["expert_reuse"] for r in rows]),
            "mi_expert_domain": st([r.get("mi_expert_task@1") for r in rows]),
            "mi_expert_class": st([r.get("mi_expert_class@1") for r in rows]),
            "seeds": [r["seed"] for r in rows],
        }
    deltas = {}
    for level, protocols in levels.items():
        learned = (protocols.get("domain_il") or {}).get("accuracy")
        oracle = (protocols.get("domain_il_oracle") or {}).get("accuracy")
        deltas[level] = {
            "routing_effect": (
                oracle["mean"] - learned["mean"] if learned and oracle else None
            )
        }
    return {"levels": levels, "deltas": deltas}


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="*", default=LEVELS)
    parser.add_argument("--seeds", default="42,1,2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument("--max_experts", type=int, default=4)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/s5b")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    study_path = os.path.join(args.out, "s5b_domain_study.json")

    paths = {
        d: build_domain_cache(
            d, os.path.join(args.out, f"cache_domain{d}"), args, device
        )
        for d in STREAM_DOMAINS
    }
    unseen_path = build_domain_cache(
        UNSEEN_DOMAIN,
        os.path.join(args.out, f"cache_domain{UNSEEN_DOMAIN}"),
        args,
        device,
    )
    stream = combine_stream(
        [paths[d] for d in STREAM_DOMAINS], os.path.join(args.out, "cache_stream")
    )

    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "rank": args.rank,
        "lambda_func": args.lambda_func,
        "levels": list(args.levels),
        "domains": STREAM_DOMAINS,
        "train_per_domain": TRAIN_PER_DOMAIN,
    }
    cells: list[dict] = []
    done: set[tuple] = set()
    if os.path.exists(study_path) and not args.force:
        previous = json.load(open(study_path))
        if previous.get("recipe") == recipe:
            cells = previous.get("cells", [])
            done = {(c["level"], c["seed"]) for c in cells}
            print(f"[S5b] resuming: {len(done)} trained cells recorded", flush=True)
        else:
            print("[S5b] different recipe, starting fresh", flush=True)

    def save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "s5b_domain_il",
            "backbone": "vit_b_16+proj768",
            "domains": {"stream": STREAM_DOMAINS, "unseen": UNSEEN_DOMAIN},
            "recipe": recipe,
            "seeds": seeds,
            "cells": cells,
            "aggregate": aggregate(cells) if cells else {},
            "contracts": {
                f"{c['level']}__{c['protocol']}__seed{c['seed']}": _contract(c)
                for c in cells
            },
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)

    for level in args.levels:
        for seed in seeds:
            if (level, seed) in done:
                continue
            args.seed = seed
            produced = run_cell(level, stream, unseen_path, args, device)
            cells.extend(produced)
            save()
            line = "  ".join(
                f"{c['protocol']}: acc={c['accuracy'] * 100:5.2f}% "
                f"unseen={c['unseen_domain_accuracy'] * 100:5.2f}%"
                for c in produced
            )
            print(f"[S5b] {level:16s} seed={seed:<3d} {line}", flush=True)

    save()
    print(f"[S5b] wrote {study_path}", flush=True)
    _print(aggregate(cells))


def _contract(cell: dict) -> dict:
    record = build_run_record(
        factors={
            "dataset": "mnist_rotated",
            "protocol": cell["protocol"],
            "task_id_at_inference": cell["task_id_at_inference"],
            "class_masking": False,
            "routing_mode": cell["routing_mode"],
            "increment_type": "domain",
            "class_space": "shared",
            "num_tasks": cell["num_tasks"],
            "classes_per_task": cell["classes_per_task"],
            "seed": cell["seed"],
            "model_family": cell["level"],
            "backbone": "vit_b_16+proj768",
            "backbone_pretraining": "imagenet_frozen",
            "readout": "ncm" if cell["level"] == "L0_ncm" else "ridge",
            "expert": (
                "none"
                if cell["level"] in ("L0_ncm", "L1_ridge")
                else "residual_adapter"
            ),
        },
        metrics={
            "learning": {
                "accuracy": cell["accuracy"],
                "forgetting": cell["forgetting"],
                "bwt": cell["bwt"],
                "fwt": cell["fwt"],
                "acc_matrix": cell["acc_matrix"],
            },
            "cost": {"stored_bytes": 0, "total_params": cell["params"]},
            "generalization": {
                "unseen_domain_accuracy": cell["unseen_domain_accuracy"]
            },
            "modular": {
                "expert_count": cell["expert_count"],
                "reuse_rate": cell["expert_reuse"],
                "routing_entropy": None,
                "utilization": None,
            },
        },
        provenance={
            "command": "experiments/s5b_domains.py",
            "mi_expert_domain": cell.get("mi_expert_task@1"),
            "mi_expert_class": cell.get("mi_expert_class@1"),
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


def _print(agg: dict) -> None:
    print("\n" + "=" * 100)
    print("S5b DOMAIN-IL (rotated MNIST, 4 stream domains + 1 unseen, frozen ViT-B/16)")
    print("=" * 100)
    print(
        f"{'level':16s} {'acc':>8s} {'forget':>8s} {'unseen':>8s} {'reuse':>7s} "
        f"{'MI(E;Dom)':>10s} {'MI(E;Cls)':>10s} {'route eff':>10s}"
    )
    for level, protocols in agg["levels"].items():
        row = protocols.get("domain_il", {})
        d = agg["deltas"][level]

        def f(x, w=8, pct=True):
            # `routing_effect` arrives as a float, the rest as {"mean", ...}.
            if x is None:
                return " " * (w - 1) + "-"
            value = x if isinstance(x, float) else x["mean"]
            v = value * 100 if pct else value
            return f"{v:{w - 1}.2f}%" if pct else f"{v:{w}.4f}"

        print(
            f"{level:16s} {f(row.get('accuracy'))} {f(row.get('forgetting'))} "
            f"{f(row.get('unseen_domain'))} {f(row.get('expert_reuse'), 7)} "
            f"{f(row.get('mi_expert_domain'), 10, False)} "
            f"{f(row.get('mi_expert_class'), 10, False)} "
            f"{f(d.get('routing_effect'), 10)}"
        )


if __name__ == "__main__":
    main()
