"""
S10 - scalability: does routing degradation come from bank capacity, candidate
count, or both?

S8 showed capacity does not close the routing tax and the memory axis only
partly does; S9 showed the router's failure mode under shift is diffuseness
rather than a confident wrong pick. S10 scales the task/expert count and asks
which of the three things that `T` moves at once the tax actually tracks:

    T up -> stored experts up        (capacity, which S8 showed does not help)
    T up -> expert count up          (bank size)
    T up -> routing candidates up    (resolution, which S8 showed partly helps)

The sweep is one variable (the partition granularity) with two constructions at
every `T`, so the S6b regime contrast is available at every point:

    contiguous   blocks of `num_classes / T` classes in label order
                 at T = 20 on CIFAR-100 this is exactly the canonical split
    dispersed    round-robin in the source's semantic order
                 at T = 20 on CIFAR-100 this is exactly S6b's `dispersed`

Two datasets, so S4b folds in as the dataset/task-count extension rather than a
separate stage: CIFAR-100 (100 classes, T in {5, 10, 20, 25}) and Tiny-ImageNet
(200 classes, T in {10, 20, 40, 50}). `classes_per_task` is recorded per cell,
because `T` changes partition granularity as well as expert count.

Per cell, beyond the S5b/S6b/S8 metrics (`L4 - L3`, `L3 - L2b`, `L4 - L2b`,
`R_iso_ncm`), S10 measures the **candidate-set decomposition** at
`m in {1, 2, 4, 8, T}`:

    coverage(m)               fraction of samples whose correct expert is in top-m
    conditional_learned(m)    accuracy on covered samples, learned top-1 route
    conditional_oracle(m)     accuracy on covered samples, oracle route
    selection_gap(m)          conditional_oracle - conditional_learned

which separates "the correct expert was not a candidate" from "it was a
candidate and the router still did not pick it". Entropy is reported
normalised (`H / log T`), because the raw maximum grows with `T`.

Guard: the T = 20 CIFAR-100 cells at seed 42 must reproduce S2's canonical
partition (`contiguous`) and S8's `dispersed` cells exactly.

Usage:
    python experiments/s10_scaling.py --device cuda
    python experiments/s10_scaling.py --datasets cifar100 --ts 5 20
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import s2_ladder  # noqa: E402
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402

DATASETS = {
    "cifar100": {
        "cache": "results/feature_cache/cifar100_vit_b16/feature_cache.pt",
        "num_classes": 100,
        "ts": [5, 10, 20, 25],
    },
    "tinyimagenet": {
        "cache": "results/s4/cache_tinyimagenet/feature_cache.pt",
        "num_classes": 200,
        "ts": [10, 20, 40, 50],
    },
}
CONSTRUCTS = ["contiguous", "dispersed"]
LEVELS = ["L0_ncm", "L1_ridge", "L2b_shared_seq", "L3_per_task", "L4_oracle"]
OPERATING = {"rank": 8, "protos": 1, "top_k": 1}
CANDIDATE_SIZES = [1, 2, 3, 4, 8]
LATENCY_BATCH = 512
LATENCY_REPEATS = 3


def _key(cell: dict) -> tuple:
    return (
        cell["dataset"],
        cell["construct"],
        cell["num_tasks"],
        cell["level"],
        cell["seed"],
    )


def default_grid(datasets, constructs, levels, seeds) -> list[dict]:
    cells = []
    for dataset in datasets:
        for num_tasks in DATASETS[dataset]["ts"]:
            for construct in constructs:
                for level in levels:
                    for seed in seeds:
                        cells.append(
                            {
                                "dataset": dataset,
                                "construct": construct,
                                "num_tasks": int(num_tasks),
                                "level": level,
                                "seed": int(seed),
                            }
                        )
    unique = {_key(c): c for c in cells}
    return list(unique.values())


# ---------------------------------------------------------------------------
# data: regroup a per-task cache into per-class features, then re-partition
# ---------------------------------------------------------------------------


def load_source(cache_path: str) -> dict:
    """All splits concatenated, so any partition can be built by masking.

    Masking (rather than concatenating per class) preserves the source sample
    order, which is what makes the T = 20 anchors exact: the same mechanism S6b
    used for its constructions.
    """
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = dict(payload["meta"])
    splits = {"train": [], "val": [], "test": []}
    for row in payload["tasks"]:
        for split in splits:
            splits[split].append(
                (row["splits"][split][0].float(), row["splits"][split][1].long())
            )
    out = {
        split: (
            torch.cat([x[0] for x in parts], dim=0),
            torch.cat([x[1] for x in parts], dim=0),
        )
        for split, parts in splits.items()
    }
    meta["num_classes"] = len(torch.unique(out["test"][1]).tolist())
    meta["classes"] = sorted(torch.unique(out["test"][1]).tolist())
    return {"meta": meta, "splits": out}


def _semantic_order(source: dict) -> list[int]:
    """The ordering `dispersed` deals out round-robin.

    For CIFAR-100 the superclass grouping is used, which is exactly S6b's
    `dispersed` at T = 20; for a dataset without a superclass axis the label
    order is used. The ordering is part of the construction's definition, not a
    free parameter.
    """
    classes = source["meta"]["classes"]
    if source["meta"].get("dataset") == "cifar100":
        import s6b_difficulty

        mapping = s6b_difficulty.superclass_of()
        return sorted(classes, key=lambda c: (mapping[c], c))
    return list(classes)


def build_tasks(source: dict, construct: str, num_tasks: int) -> list[dict]:
    """One construction: a regrouping of class ids into `num_tasks` tasks.

    Built by masking the concatenated source splits, so the sample order inside
    a task is the source order - identical to S6b's construction code, which is
    what the T = 20 anchors check.
    """
    classes = source["meta"]["classes"]
    if len(classes) % num_tasks != 0:
        raise ValueError(f"{len(classes)} classes do not divide into {num_tasks}")
    per_task = len(classes) // num_tasks

    if construct == "contiguous":
        groups = [classes[i * per_task : (i + 1) * per_task] for i in range(num_tasks)]
    elif construct == "dispersed":
        groups = [[] for _ in range(num_tasks)]
        for index, c in enumerate(_semantic_order(source)):
            groups[index % num_tasks].append(c)
    else:
        raise KeyError(f"unknown construct {construct!r}")

    tasks = []
    for task_id, group in enumerate(groups):
        assert len(group) == per_task, (task_id, group)
        keep = torch.tensor([int(c) for c in group])
        splits = {}
        for split, (feats, labels) in source["splits"].items():
            mask = torch.isin(labels, keep)
            splits[split] = (feats[mask], labels[mask])
        tasks.append(
            {
                "task_id": task_id,
                "classes": sorted(int(c) for c in group),
                "splits": splits,
            }
        )
    return tasks


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def train_model(level: str, train_tasks: list[dict], args, device):
    dim = int(train_tasks[0]["splits"]["train"][0].size(1))
    num_classes = sum(len(t["classes"]) for t in train_tasks)
    spec = s2_ladder.LEVELS_BY_NAME[level]
    cell_args = argparse.Namespace(
        rank=OPERATING["rank"],
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        lambda_func=args.lambda_func,
        max_experts=max(20, len(train_tasks)),
    )
    s2_ladder.set_seed(args.seed)
    model = s2_ladder.LadderModel(
        spec, dim, num_classes, cell_args, device, router_slots=num_classes
    )
    model.router_prototypes = OPERATING["protos"]
    model.top_k_experts = OPERATING["top_k"]
    for t, task in enumerate(train_tasks):
        model.seen = sorted(set(model.seen) | set(task["classes"]))
        model.fit_task(task, t)
        model.register_task(task, t)
    return model


@torch.no_grad()
def evaluate(model, level: str, test_tasks: list[dict]) -> dict:
    spec = s2_ladder.LEVELS_BY_NAME[level]
    oracle = spec.router == "oracle"
    n = len(test_tasks)
    R = np.zeros((n, n), dtype=np.float32)
    for t in range(n):
        for i in range(t + 1):
            R[t, i] = model.evaluate_task(test_tasks[i], oracle=oracle, task_id=i)
    T = n - 1
    return {
        "accuracy": float(np.mean(R[T, :])),
        "retention": float(np.mean(R[T, :T])) if T > 0 else 0.0,
        "acc_matrix": R.tolist(),
    }


@torch.no_grad()
def routing_report(model, test_tasks: list[dict]) -> dict:
    """Routing metrics plus the candidate-set coverage/selection decomposition."""
    if model.router is None or not model.experts or model.spec.router == "oracle":
        return {}
    n_experts = len(model.experts)
    sizes = sorted({min(m, n_experts) for m in CANDIDATE_SIZES + [n_experts]})
    totals = 0
    covered_learned = {m: 0 for m in sizes}
    covered_oracle = {m: 0 for m in sizes}
    covered_n = {m: 0 for m in sizes}
    shares = []
    entropies = []
    for task in test_tasks:
        feats, labels = task["splits"]["test"]
        feats, labels = feats.to(model.device), labels.to(model.device)
        task_id = int(task["task_id"])
        ids, _ = model.router.top_k(feats, k=max(sizes))
        learned = model.logits(feats).argmax(dim=-1) == labels
        oracle = (
            model.logits(feats, oracle=True, task_id=task_id).argmax(dim=-1) == labels
        )
        totals += feats.size(0)
        for m in sizes:
            covered = (ids[:, :m] == task_id).any(dim=-1)
            covered_n[m] += int(covered.sum())
            covered_learned[m] += int(learned[covered].sum())
            covered_oracle[m] += int(oracle[covered].sum())
        top1 = ids[:, 0]
        share = torch.bincount(top1, minlength=n_experts).float() / max(top1.numel(), 1)
        shares.append(float(share.max()))
        p = share[share > 0]
        entropies.append(float(-(p * p.log()).sum()))
    report = {
        "task_recall_at_1": None,
        "task_recall_at_3": None,
        "expert_max_share": float(np.mean(shares)) if shares else None,
        "expert_entropy": float(np.mean(entropies)) if entropies else None,
        "expert_entropy_normalized": (
            float(np.mean(entropies) / np.log(max(n_experts, 2))) if entropies else None
        ),
        "coverage": {},
        "conditional_learned": {},
        "conditional_oracle": {},
        "selection_gap": {},
    }
    for m in sizes:
        coverage = covered_n[m] / max(totals, 1)
        report["coverage"][str(m)] = coverage
        report["conditional_learned"][str(m)] = covered_learned[m] / max(
            covered_n[m], 1
        )
        report["conditional_oracle"][str(m)] = covered_oracle[m] / max(covered_n[m], 1)
        report["selection_gap"][str(m)] = (
            report["conditional_oracle"][str(m)] - report["conditional_learned"][str(m)]
        )
    report["task_recall_at_1"] = report["coverage"].get("1")
    report["task_recall_at_3"] = report["coverage"].get("3")
    return report


@torch.no_grad()
def measure_latency(
    model, task: dict, level: str, repeats: int = LATENCY_REPEATS
) -> float:
    """Median ms per sample over `repeats` forward passes on a fixed batch."""
    feats = task["splits"]["test"][0][:LATENCY_BATCH].to(model.device)
    if feats.size(0) == 0:
        return float("nan")
    oracle = s2_ladder.LEVELS_BY_NAME[level].router == "oracle"
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        model.logits(feats, oracle=oracle, task_id=int(task["task_id"]))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - started) / feats.size(0) * 1000.0)
    return float(statistics.median(times))


def run_cell(cell, args, device, store_cache) -> dict:
    dataset = cell["dataset"]
    if dataset not in store_cache:
        store_cache[dataset] = load_source(DATASETS[dataset]["cache"])
    tasks = build_tasks(store_cache[dataset], cell["construct"], cell["num_tasks"])
    args.seed = cell["seed"]
    model = train_model(cell["level"], tasks, args, device)
    result = evaluate(model, cell["level"], tasks)
    return {
        **cell,
        **result,
        "classes_per_task": len(tasks[0]["classes"]),
        "num_classes": sum(len(t["classes"]) for t in tasks),
        "routing": routing_report(model, tasks),
        "resources": model.resources(),
        "latency_ms_per_sample": measure_latency(model, tasks[0], cell["level"]),
    }


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def aggregate(cells: list[dict], datasets, constructs, levels) -> dict:
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

    def pick(**kw):
        return [c for c in cells if all(c[k] == v for k, v in kw.items())]

    out: dict[str, list] = {}
    for dataset in datasets:
        rows = []
        for num_tasks in sorted(
            {c["num_tasks"] for c in cells if c["dataset"] == dataset}
        ):
            for construct in constructs:
                point = dict(dataset=dataset, construct=construct, num_tasks=num_tasks)
                sel = pick(**point)
                if not sel:
                    continue
                by_level = {
                    level: st([r["accuracy"] for r in sel if r["level"] == level])
                    for level in levels
                }
                l0 = by_level.get("L0_ncm")
                l2b = by_level.get("L2b_shared_seq")
                l3 = by_level.get("L3_per_task")
                l4 = by_level.get("L4_oracle")
                entry = {
                    "dataset": dataset,
                    "construct": construct,
                    "num_tasks": num_tasks,
                    "classes_per_task": sel[0]["classes_per_task"],
                    "levels": by_level,
                    "n": len(sel),
                }
                if l3 and l4:
                    entry["routing_tax"] = l4["mean"] - l3["mean"]
                if l3 and l2b:
                    entry["isolation_realised"] = l3["mean"] - l2b["mean"]
                if l4 and l2b:
                    entry["isolation_available"] = l4["mean"] - l2b["mean"]
                if l3 and l0 and l4 and (l4["mean"] - l0["mean"]) > 1e-9:
                    entry["R_iso_ncm"] = (l3["mean"] - l0["mean"]) / (
                        l4["mean"] - l0["mean"]
                    )
                if l3 and l2b and l4 and (l4["mean"] - l2b["mean"]) > 1e-9:
                    entry["R_iso"] = (l3["mean"] - l2b["mean"]) / (
                        l4["mean"] - l2b["mean"]
                    )
                l3_rows = [r for r in sel if r["level"] == "L3_per_task"]
                if l3_rows:
                    entry["routing"] = {
                        key: st([r["routing"].get(key) for r in l3_rows])
                        for key in (
                            "task_recall_at_1",
                            "task_recall_at_3",
                            "expert_max_share",
                            "expert_entropy_normalized",
                        )
                    }
                    entry["coverage"] = {
                        m: st(
                            [r["routing"].get("coverage", {}).get(m) for r in l3_rows]
                        )
                        for m in sorted(l3_rows[0]["routing"].get("coverage", {}))
                    }
                    entry["selection_gap"] = {
                        m: st(
                            [
                                r["routing"].get("selection_gap", {}).get(m)
                                for r in l3_rows
                            ]
                        )
                        for m in sorted(l3_rows[0]["routing"].get("selection_gap", {}))
                    }
                    entry["conditional_oracle"] = {
                        m: st(
                            [
                                r["routing"].get("conditional_oracle", {}).get(m)
                                for r in l3_rows
                            ]
                        )
                        for m in sorted(
                            l3_rows[0]["routing"].get("conditional_oracle", {})
                        )
                    }
                    entry["resources"] = l3_rows[0]["resources"]
                    entry["latency_ms_per_sample"] = st(
                        [r["latency_ms_per_sample"] for r in l3_rows]
                    )
                rows.append(entry)
        out[dataset] = rows
    return {"scaling": out}


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def verify_against_anchors(cells: list[dict]) -> dict:
    """T = 20 on CIFAR-100 must reproduce the stages it continues.

    `contiguous` at T = 20 is the canonical split, whose seed-42 values are S2's
    (`results/s2/s2_ladder_study.json`, the oldest ladder run); `dispersed` at
    T = 20 is S6b's construction, whose seed-42 values are S8's.
    """
    out: dict = {}
    s2_path = "results/s2/s2_ladder_study.json"
    if os.path.exists(s2_path):
        s2 = {
            r["level"]: r["avg_accuracy"]
            for r in json.load(open(s2_path))["records"]
            if r["seed"] == 42
        }
        here = {
            c["level"]: c["accuracy"]
            for c in cells
            if c["dataset"] == "cifar100"
            and c["construct"] == "contiguous"
            and c["num_tasks"] == 20
            and c["seed"] == 42
        }
        matched = sorted(set(s2) & set(here))
        out["s2_contiguous_t20"] = {
            "checked": len(matched),
            "max_abs_delta": (
                max(abs(s2[k] - here[k]) for k in matched) if matched else None
            ),
        }
    s8_path = "results/s8/s8_budget_study.json"
    if os.path.exists(s8_path):
        s8 = {
            c["level"]: c["accuracy"]
            for c in json.load(open(s8_path))["cells"]
            if c["construct"] == "dispersed"
            and c["rank"] == OPERATING["rank"]
            and c["protos"] == OPERATING["protos"]
            and c["top_k"] == OPERATING["top_k"]
            and c["seed"] == 42
        }
        here = {
            c["level"]: c["accuracy"]
            for c in cells
            if c["dataset"] == "cifar100"
            and c["construct"] == "dispersed"
            and c["num_tasks"] == 20
            and c["seed"] == 42
        }
        matched = sorted(set(s8) & set(here))
        out["s8_dispersed_t20"] = {
            "checked": len(matched),
            "max_abs_delta": (
                max(abs(s8[k] - here[k]) for k in matched) if matched else None
            ),
        }
    return out


def _contract(cell: dict) -> dict:
    spec = s2_ladder.LEVELS_BY_NAME[cell["level"]]
    resources = cell["resources"]
    record = build_run_record(
        factors={
            "dataset": cell["dataset"],
            "protocol": "class_il",
            "task_id_at_inference": cell["level"] == "L4_oracle",
            "class_masking": False,
            "routing_mode": "oracle" if cell["level"] == "L4_oracle" else "learned",
            "increment_type": "class",
            "class_space": "task",
            "num_tasks": cell["num_tasks"],
            "classes_per_task": cell["classes_per_task"],
            "seed": cell["seed"],
            "task_order_seed": None,
            "model_family": cell["level"],
            "backbone": "vit_b_16+proj768",
            "backbone_pretraining": "imagenet_frozen",
            "readout": spec.readout,
            "expert": spec.expert or "none",
        },
        metrics={
            "learning": {
                "accuracy": cell["accuracy"],
                "forgetting": 0.0,
                "acc_matrix": cell["acc_matrix"],
            },
            "cost": {
                "stored_bytes": resources["memory_bytes"],
                "total_params": resources["total_params"],
            },
        },
        provenance={
            "command": "experiments/s10_scaling.py",
            "construct": cell["construct"],
            "num_classes": cell["num_classes"],
            "retention": cell["retention"],
            "routing": cell["routing"],
            "resources": resources,
            "latency_ms_per_sample": cell["latency_ms_per_sample"],
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


def _print(agg: dict) -> None:
    def pct(stats):
        return f"{stats['mean'] * 100:6.2f}" if stats else "   n/a"

    def num(value, digits=3):
        return f"{value:.{digits}f}" if value is not None else "n/a"

    print("\n" + "=" * 112)
    print("S10 SCALABILITY (frozen ViT-B/16, Class-IL, rank 8)")
    print("=" * 112)
    for dataset, rows in agg["scaling"].items():
        print(f"\n########## {dataset}")
        print(
            "| construct | T | cls/task | L0 | L1 | L2b | L3 | L4 | tax | "
            "realised | available | R_iso_ncm | recall@1 | recall@3 | max-share | H_norm | "
            "latency ms | stored |"
        )
        print(
            "| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | "
            "--: | --: | --: | --: | --: | --: | --: |"
        )
        for row in rows:
            lv = row["levels"]
            routing = row.get("routing", {})
            resources = row.get("resources", {}) or {}
            print(
                f"| {row['construct']} | {row['num_tasks']} | {row['classes_per_task']} | "
                f"{pct(lv.get('L0_ncm'))} | {pct(lv.get('L1_ridge'))} | "
                f"{pct(lv.get('L2b_shared_seq'))} | "
                f"{pct(lv.get('L3_per_task'))} | {pct(lv.get('L4_oracle'))} | "
                f"{num(row.get('routing_tax') and row['routing_tax'] * 100, 2)} | "
                f"{num(row.get('isolation_realised') and row['isolation_realised'] * 100, 2)} | "
                f"{num(row.get('isolation_available') and row['isolation_available'] * 100, 2)} | "
                f"{num(row.get('R_iso_ncm'))} | "
                f"{num((routing.get('task_recall_at_1') or {}).get('mean'))} | "
                f"{num((routing.get('task_recall_at_3') or {}).get('mean'))} | "
                f"{num((routing.get('expert_max_share') or {}).get('mean'))} | "
                f"{num((routing.get('expert_entropy_normalized') or {}).get('mean'))} | "
                f"{num((row.get('latency_ms_per_sample') or {}).get('mean'), 4)} | "
                f"{resources.get('memory_bytes', 0) / 1024:.1f} KiB |"
            )
        print("\n  candidate-set decomposition (L3): coverage / selection gap")
        print(
            f"  {'construct':10s} {'T':>3} {'m':>4} {'coverage':>9} {'sel_gap':>8} {'cond_oracle':>12}"
        )
        for row in rows:
            coverage = row.get("coverage", {})
            gap = row.get("selection_gap", {})
            cond = row.get("conditional_oracle", {})
            for m in sorted(coverage, key=int):
                print(
                    f"  {row['construct']:10s} {row['num_tasks']:>3} {m:>4} "
                    f"{num((coverage[m] or {}).get('mean')):>9} "
                    f"{num((gap.get(m) or {}).get('mean')):>8} "
                    f"{num((cond.get(m) or {}).get('mean')):>12}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--constructs", nargs="*", default=CONSTRUCTS)
    parser.add_argument("--ts", nargs="*", type=int, default=None)
    parser.add_argument("--levels", nargs="*", default=LEVELS)
    parser.add_argument("--seeds", default="42,1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/s10")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.ts:
        for dataset in args.datasets:
            DATASETS[dataset]["ts"] = [
                t for t in args.ts if DATASETS[dataset]["num_classes"] % t == 0
            ]
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    study_path = os.path.join(args.out, "s10_scaling_study.json")

    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_func": args.lambda_func,
        "datasets": {
            name: {"num_classes": spec["num_classes"], "ts": spec["ts"]}
            for name, spec in DATASETS.items()
            if name in args.datasets
        },
        "constructs": list(args.constructs),
        "levels": list(args.levels),
        "operating": OPERATING,
        "candidate_sizes": CANDIDATE_SIZES,
    }
    cells: list[dict] = []
    done: set[tuple] = set()
    if os.path.exists(study_path) and not args.force:
        previous = json.load(open(study_path))
        if previous.get("recipe") == recipe:
            cells = previous.get("cells", [])
            done = {_key(c) for c in cells}
            print(f"[S10] resuming: {len(done)} cells recorded", flush=True)

    def save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "s10_scalability",
            "backbone": "vit_b_16+proj768",
            "recipe": recipe,
            "seeds": seeds,
            "cells": cells,
            "aggregate": (
                aggregate(cells, args.datasets, args.constructs, args.levels)
                if cells
                else {}
            ),
            "contracts": {
                "__".join(str(part) for part in _key(c)): _contract(c) for c in cells
            },
            "anchors": verify_against_anchors(cells),
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)

    grid = default_grid(args.datasets, args.constructs, args.levels, seeds)
    print(f"[S10] grid: {len(grid)} cells, {len(done)} already done", flush=True)

    store_cache: dict[str, dict] = {}
    for cell in grid:
        if _key(cell) in done:
            continue
        result = run_cell(cell, args, device, store_cache)
        cells.append(result)
        done.add(_key(cell))
        save()
        routing = result["routing"]
        print(
            f"[S10] {cell['dataset']:13s} {cell['construct']:10s} T={cell['num_tasks']:<3d} "
            f"{cell['level']:16s} seed={cell['seed']:<3d} "
            f"acc={result['accuracy'] * 100:6.2f}%  "
            f"recall@1={(routing.get('task_recall_at_1') or float('nan')):.3f}  "
            f"share={(routing.get('expert_max_share') or float('nan')):.3f}  "
            f"mem={result['resources']['memory_bytes'] / 1024:8.1f}KiB",
            flush=True,
        )

    save()
    print(f"[S10] wrote {study_path}", flush=True)
    payload = json.load(open(study_path))
    _print(payload["aggregate"])
    print(f"\n[S10 anchors] {json.dumps(payload['anchors'], indent=1)}")


if __name__ == "__main__":
    main()
