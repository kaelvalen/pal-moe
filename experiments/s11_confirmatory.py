"""
S11 - the confirmatory stage: pre-registered claims, paired over seeds.

Every hypothesis here was derived from an earlier stage; S11 does not explore,
it tests. The design is paired by construction - the same feature cache, the
same partition and the same seed produce `L4 - L3`, `L3 - L2b` and `L1 - L3` as
*differences within a seed*, not as independent samples - so the report shows the
per-seed difference distribution first and the summary statistics after.

Six seeds, not five, for a statistical reason: with N = 5 the exact two-sided
sign test cannot reach p < 0.05 at all (its smallest value is 2 * 0.5^5 =
0.0625); with N = 6 it can (0.0312). Small-sample inference therefore uses exact
tests - a sign test and an exact paired permutation test over all 2^N sign
flips - with a Holm correction over the primary family, and a t-based interval
only as a summary. No asymptotic p-value is relied on.

The pre-registered hypotheses:

    H1  under the Stage-1 operating point (frozen ViT-B/16, Class-IL, rank 8,
        one prototype, top_k = 1) L1 Ridge stays competitive with or superior
        to the learned expert variants while requiring no gradient optimization
    H2  L4 - L3 is a systematic routing tax, not seed noise
    H3  the tax is larger under `dispersed` than under `coherent` (cross-task
        overlap), paired by seed
    H4  capacity growth does not reduce the tax, while routing resolution does:
        the rank axis moves it less than the prototype axis, paired
    H5  routing resolution helps but does not eliminate the tax
    H6  `R_iso` and `R_iso_ncm` measure different things: the shared-sequential
        denominator overstates realization, so `R_iso > R_iso_ncm`

Nothing new is added to the model: no router, no rejection head, no loss, no
merge rule. The stage only measures the existing ladder more carefully.

Usage:
    python experiments/s11_confirmatory.py --device cuda
"""

import argparse
import itertools
import json
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import s2_ladder  # noqa: E402
import s6b_difficulty  # noqa: E402
import s10_scaling  # noqa: E402
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402

SOURCE_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"
TINY_CACHE = "results/s4/cache_tinyimagenet/feature_cache.pt"
CONSTRUCTS = ["coherent", "dispersed"]
OPERATING = {"rank": 8, "protos": 1, "top_k": 1}
RANKS = [2, 8, 32, 128]
PROTOTYPES = [4, 16]
SCALE_T = [5, 25]
SESOI_TAX_POINTS = 2.0  # smallest capacity effect on the tax we would call real


def default_grid(constructs, seeds, levels=None) -> list[dict]:
    """Four parts, each answering one hypothesis, one variable at a time."""
    cells: list[dict] = []

    def add(part, construct, level, rank, protos, num_tasks, seed, dataset="cifar100"):
        cells.append(
            {
                "part": part,
                "dataset": dataset,
                "construct": construct,
                "level": level,
                "rank": int(rank),
                "protos": int(protos),
                "top_k": OPERATING["top_k"],
                "num_tasks": int(num_tasks),
                "seed": int(seed),
            }
        )

    for construct in constructs:
        for seed in seeds:
            # A: the training-free references at the operating point (H1)
            for level in ("L0_ncm", "L1_ridge"):
                add("A", construct, level, OPERATING["rank"], 1, 20, seed)
            # B: the parameter axis (H2, H4) and the isolation decomposition
            for rank in RANKS:
                for level in ("L2b_shared_seq", "L3_per_task", "L4_oracle"):
                    add("B", construct, level, rank, 1, 20, seed)
            # C: the memory axis (H5)
            for protos in PROTOTYPES:
                add("C", construct, "L3_per_task", OPERATING["rank"], protos, 20, seed)
    # E: the scale chain (the S10 result, on more seeds at its endpoints)
    for seed in seeds:
        for num_tasks in SCALE_T:
            for level in ("L2b_shared_seq", "L3_per_task", "L4_oracle"):
                add("E", "dispersed", level, OPERATING["rank"], 1, num_tasks, seed)
    unique = {_key(c): c for c in cells}
    out = list(unique.values())
    if levels:
        out = [c for c in out if c["level"] in levels]
    return out


def _key(cell: dict) -> tuple:
    return (
        cell["part"],
        cell["dataset"],
        cell["construct"],
        cell["level"],
        cell["rank"],
        cell["protos"],
        cell["num_tasks"],
        cell["seed"],
    )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _base_tasks(cell: dict, source_cache: dict):
    """Tasks for a cell: S6b's construction at T = 20, S10's partition otherwise.

    Both paths exist because S6b's construction and S10's partition builder must
    each be the single source of truth for their own regime; the anchors check
    that the two agree at T = 20 on CIFAR-100.
    """
    dataset = cell["dataset"]
    entry = source_cache.setdefault(dataset, {"tasks": None, "store": None})
    if cell["num_tasks"] == 20 and dataset == "cifar100":
        if entry["tasks"] is None:
            _, tasks = s2_ladder.load_tasks(SOURCE_CACHE)
            entry["tasks"] = {
                construct: s6b_difficulty.build_construction(tasks, construct)
                for construct in CONSTRUCTS
            }
        return entry["tasks"][cell["construct"]]
    if entry["store"] is None:
        path = SOURCE_CACHE if dataset == "cifar100" else TINY_CACHE
        entry["store"] = s10_scaling.load_source(path)
    return s10_scaling.build_tasks(entry["store"], cell["construct"], cell["num_tasks"])


def train_model(level: str, tasks: list[dict], cell: dict, args, device):
    dim = int(tasks[0]["splits"]["train"][0].size(1))
    num_classes = sum(len(t["classes"]) for t in tasks)
    spec = s2_ladder.LEVELS_BY_NAME[level]
    cell_args = argparse.Namespace(
        rank=cell["rank"],
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=cell["seed"],
        lambda_func=args.lambda_func,
        max_experts=max(20, len(tasks)),
    )
    s2_ladder.set_seed(cell["seed"])
    model = s2_ladder.LadderModel(
        spec, dim, num_classes, cell_args, device, router_slots=num_classes
    )
    model.router_prototypes = cell["protos"]
    model.top_k_experts = cell["top_k"]
    for t, task in enumerate(tasks):
        model.seen = sorted(set(model.seen) | set(task["classes"]))
        model.fit_task(task, t)
        model.register_task(task, t)
    return model


@torch.no_grad()
def evaluate(model, level: str, tasks: list[dict]) -> dict:
    spec = s2_ladder.LEVELS_BY_NAME[level]
    oracle = spec.router == "oracle"
    n = len(tasks)
    R = np.zeros((n, n), dtype=np.float32)
    for t in range(n):
        for i in range(t + 1):
            R[t, i] = model.evaluate_task(tasks[i], oracle=oracle, task_id=i)
    T = n - 1
    return {
        "accuracy": float(np.mean(R[T, :])),
        "retention": float(np.mean(R[T, :T])) if T > 0 else 0.0,
        "acc_matrix": R.tolist(),
        "routing": model.routing_stats(tasks, k=3),
        "resources": model.resources(),
    }


def run_cell(cell: dict, args, device, source_cache: dict) -> dict:
    tasks = _base_tasks(cell, source_cache)
    args.seed = cell["seed"]
    model = train_model(cell["level"], tasks, cell, args, device)
    result = evaluate(model, cell["level"], tasks)
    return {
        **cell,
        **result,
        "classes_per_task": len(tasks[0]["classes"]),
        "num_classes": sum(len(t["classes"]) for t in tasks),
    }


# ---------------------------------------------------------------------------
# small-sample statistics
# ---------------------------------------------------------------------------


def paired_stats(values: list[float], label: str, sesoi: float | None = None) -> dict:
    """Exact small-sample inference on N paired differences.

    Reports the per-seed distribution first, then mean/SD and a t-based interval
    as a summary, and two *exact* tests: a sign test (only defined away from
    zero) and a paired permutation test over all 2^N sign flips, which is exact
    for any N and handles ties.
    """
    clean = [v for v in values if v is not None]
    n = len(clean)
    if n == 0:
        return {"label": label, "n": 0}
    mean = statistics.mean(clean)
    sd = statistics.stdev(clean) if n > 1 else 0.0
    # t-based 95% interval, df = n - 1; a small table rather than a dependency.
    t95 = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }.get(n - 1, 1.96)
    half = t95 * sd / math.sqrt(n) if n > 1 else 0.0

    positive = sum(1 for v in clean if v > 0)
    negative = sum(1 for v in clean if v < 0)
    nonzero = positive + negative
    sign_p = None
    if nonzero:
        k = min(positive, negative)
        sign_p = min(
            1.0, 2 * sum(math.comb(nonzero, i) for i in range(k + 1)) / 2**nonzero
        )

    # exact paired permutation: every sign flip of the observed differences
    observed = abs(mean)
    extreme = 0
    total = 0
    for flips in itertools.product((1, -1), repeat=n):
        permuted = abs(statistics.mean([f * v for f, v in zip(flips, clean)]))
        extreme += permuted >= observed - 1e-12
        total += 1
    permutation_p = extreme / total if total else None

    out = {
        "label": label,
        "n": n,
        "per_seed": clean,
        "mean": mean,
        "sd": sd,
        "ci95": [mean - half, mean + half],
        "positive": positive,
        "negative": negative,
        "sign_p": sign_p,
        "permutation_p": permutation_p,
    }
    if sesoi is not None:
        out["sesoi"] = sesoi
        out["within_sesoi"] = bool(abs(mean) + half <= sesoi)
    return out


def signed_rank_statistic(values: list[float]) -> float:
    """Two-sided Wilcoxon statistic: `max(W+, W-)` over the signed ranks.

    Used as the per-test statistic for the max-statistic correction because it is
    scale-free and comparable across tests (unlike a raw mean, which would let
    the largest-scale test dominate) and non-degenerate (unlike a t-statistic,
    which is infinite when the differences are perfectly consistent).

    It must be two-sided: `W+` alone scores a perfectly consistent *negative*
    effect as zero, which made the first run of this stage report the memory
    axis as "not rejected" while every seed agreed on its sign.
    """
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = float(rank)
    positive = sum(ranks[i] for i in range(len(values)) if values[i] > 0)
    negative = sum(ranks[i] for i in range(len(values)) if values[i] < 0)
    return max(positive, negative)


def westfall_young(tests: dict[str, list[float]], n_seeds: int) -> dict:
    """Single-step max-statistic (Westfall-Young) correction over a family.

    The tests are paired on the same seeds, so one shared sign-flip permutation
    scheme gives the *joint* null distribution, and the correlation between the
    tests is accounted for instead of paid as a factor of `m`. This matters at
    N = 6: Holm needs p <= alpha/m, and the smallest achievable permutation
    p-value at six seeds is 2/64 = 0.031, so Holm can never reject for any family
    of two or more tests. The max-statistic can.
    """
    names = [
        n for n, v in tests.items() if len(v) == n_seeds and any(v != 0 for v in v)
    ]
    if not names:
        return {}
    observed = {n: signed_rank_statistic(tests[n]) for n in names}
    maxima = []
    for flips in itertools.product((1, -1), repeat=n_seeds):
        flipped = {
            n: signed_rank_statistic([f * v for f, v in zip(flips, tests[n])])
            for n in names
        }
        maxima.append(max(flipped.values()))
    out = {}
    for n in names:
        adjusted = sum(1 for m in maxima if m >= observed[n] - 1e-12) / len(maxima)
        out[n] = {"raw_statistic": observed[n], "adjusted_p": adjusted}
    return out


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni over the primary family, monotone-adjusted."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running = 0.0
    for index, (name, p) in enumerate(items):
        value = min(1.0, (m - index) * p)
        running = max(running, value)
        adjusted[name] = {"raw_p": p, "adjusted_p": running, "reject": running < alpha}
    return adjusted


def order_variance(path: str = "results/s6/s6_order_study.json") -> dict:
    """The order-variance component, read from S6 rather than re-run.

    S6 measured the tax across nine order configurations (three class orders x
    three task orders) at the same operating point S11 uses (rank 8, ten
    epochs), so its cells are the order-variance arm of this stage.

    S6 holds one task out, so the evaluated class set depends on the class order
    and even L0/L1 move across configurations; their spread is therefore the
    baseline against which the tax's spread is read, not a zero check. A tax
    spread *smaller* than the accuracy spreads is the statement.
    """
    if not os.path.exists(path):
        return {"note": "S6 study not found"}
    cells = json.load(open(path))["cells"]
    by_level: dict[str, list[float]] = {}
    for cell in cells:
        by_level.setdefault(cell["level"], []).append(cell["accuracy"])

    def spread(values):
        return {
            "n": len(values),
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    taxes = []
    for class_seed in (0, 1, 2):
        for task_seed in (0, 1, 2):
            l3 = next(
                (
                    c["accuracy"]
                    for c in cells
                    if c["class_order_seed"] == class_seed
                    and c["task_order_seed"] == task_seed
                    and c["level"] == "L3_per_task"
                ),
                None,
            )
            l4 = next(
                (
                    c["accuracy"]
                    for c in cells
                    if c["class_order_seed"] == class_seed
                    and c["task_order_seed"] == task_seed
                    and c["level"] == "L4_oracle"
                ),
                None,
            )
            if l3 is not None and l4 is not None:
                taxes.append(l4 - l3)
    return {
        "source": path,
        "levels": {level: spread(values) for level, values in by_level.items()},
        "tax_across_orders": spread(taxes),
        "note": "three class orders x three task orders; one task is held out, "
        "so the level spreads are the baseline for the tax spread",
    }


def _cells_index(cells: list[dict]) -> dict:
    return {
        (
            c["construct"],
            c["level"],
            c["rank"],
            c["protos"],
            c["num_tasks"],
            c["seed"],
        ): c
        for c in cells
    }


def _acc(index: dict, construct, level, seed, rank=8, protos=1, num_tasks=20):
    cell = index.get((construct, level, rank, protos, num_tasks, seed))
    return cell["accuracy"] if cell else None


def build_hypotheses(cells: list[dict], seeds: list[int]) -> dict:
    """The pre-registered tests, all paired over seeds."""
    index = _cells_index(cells)
    out: dict = {"primary": {}, "secondary": {}}

    # H1: L1 Ridge versus the learned expert variants at the operating point
    for construct in CONSTRUCTS:
        diffs = [
            (
                _acc(index, construct, "L1_ridge", seed),
                _acc(index, construct, "L3_per_task", seed),
            )
            for seed in seeds
        ]
        values = [a - b for a, b in diffs if a is not None and b is not None]
        out["primary"][f"H1_{construct}"] = paired_stats(
            values, f"H1: L1_ridge - L3_per_task ({construct})"
        )

    # H2: the routing tax, per regime
    for construct in CONSTRUCTS:
        values = []
        for seed in seeds:
            l3 = _acc(index, construct, "L3_per_task", seed)
            l4 = _acc(index, construct, "L4_oracle", seed)
            if l3 is not None and l4 is not None:
                values.append(l4 - l3)
        out["primary"][f"H2_{construct}"] = paired_stats(
            values, f"H2: L4 - L3 ({construct})"
        )

    # H3: the tax is larger under dispersed, paired by seed
    values = []
    for seed in seeds:
        coh3 = _acc(index, "coherent", "L3_per_task", seed)
        coh4 = _acc(index, "coherent", "L4_oracle", seed)
        dis3 = _acc(index, "dispersed", "L3_per_task", seed)
        dis4 = _acc(index, "dispersed", "L4_oracle", seed)
        if None not in (coh3, coh4, dis3, dis4):
            values.append((dis4 - dis3) - (coh4 - coh3))
    out["primary"]["H3_overlap"] = paired_stats(
        values, "H3: tax(dispersed) - tax(coherent)"
    )

    # H4: the capacity axis moves the tax less than the resolution axis
    for construct in CONSTRUCTS:
        rank_values = []
        for seed in seeds:
            low3 = _acc(index, construct, "L3_per_task", seed, rank=RANKS[0])
            low4 = _acc(index, construct, "L4_oracle", seed, rank=RANKS[0])
            high3 = _acc(index, construct, "L3_per_task", seed, rank=RANKS[-1])
            high4 = _acc(index, construct, "L4_oracle", seed, rank=RANKS[-1])
            if None not in (low3, low4, high3, high4):
                rank_values.append((high4 - high3) - (low4 - low3))
        out["secondary"][f"H4_rank_effect_{construct}"] = paired_stats(
            rank_values,
            f"H4: tax(rank {RANKS[-1]}) - tax(rank {RANKS[0]}) ({construct})",
            sesoi=SESOI_TAX_POINTS / 100.0,
        )
        mem_values = []
        for seed in seeds:
            base3 = _acc(index, construct, "L3_per_task", seed, protos=1)
            base4 = _acc(index, construct, "L4_oracle", seed, protos=1)
            rich3 = _acc(index, construct, "L3_per_task", seed, protos=PROTOTYPES[-1])
            if None not in (base3, base4, rich3):
                # tax(protos 1) - tax(protos 16); L4 does not depend on the
                # router's prototype budget, so the tax difference is just the
                # L3 difference.
                mem_values.append(rich3 - base3)
        out["primary"][f"H5_memory_effect_{construct}"] = paired_stats(
            mem_values,
            f"H5: tax(protos 1) - tax(protos {PROTOTYPES[-1]}) ({construct})",
        )

    # H5 (continued): resolution helps but does not eliminate
    for construct in CONSTRUCTS:
        values = []
        for seed in seeds:
            rich3 = _acc(index, construct, "L3_per_task", seed, protos=PROTOTYPES[-1])
            rich4 = _acc(index, construct, "L4_oracle", seed, protos=1)
            if rich3 is not None and rich4 is not None:
                values.append(rich4 - rich3)
        out["secondary"][f"H5_residual_tax_{construct}"] = paired_stats(
            values, f"H5 residual: tax after {PROTOTYPES[-1]} prototypes ({construct})"
        )

    # H6: R_iso and R_iso_ncm measure different things
    for construct in CONSTRUCTS:
        values = []
        for seed in seeds:
            l0 = _acc(index, construct, "L0_ncm", seed)
            l1 = _acc(index, construct, "L1_ridge", seed)
            l2b = _acc(index, construct, "L2b_shared_seq", seed)
            l3 = _acc(index, construct, "L3_per_task", seed)
            l4 = _acc(index, construct, "L4_oracle", seed)
            if None in (l0, l1, l2b, l3, l4):
                continue
            available = l4 - l2b
            realised = l3 - l2b
            if available <= 0 or (l4 - l0) <= 0:
                continue
            values.append(realised / available - (l3 - l0) / (l4 - l0))
        out["primary"][f"H6_ratio_gap_{construct}"] = paired_stats(
            values, f"H6: R_iso - R_iso_ncm ({construct})"
        )

    # H4 (continued): the equivalence view, capacity against a pre-registered SESOI
    for construct in CONSTRUCTS:
        values = []
        for seed in seeds:
            low3 = _acc(index, construct, "L3_per_task", seed, rank=RANKS[0])
            low4 = _acc(index, construct, "L4_oracle", seed, rank=RANKS[0])
            high3 = _acc(index, construct, "L3_per_task", seed, rank=RANKS[-1])
            high4 = _acc(index, construct, "L4_oracle", seed, rank=RANKS[-1])
            if None not in (low3, low4, high3, high4):
                values.append((high4 - high3) - (low4 - low3))
        out["secondary"][f"H4_equivalence_{construct}"] = paired_stats(
            values,
            f"H4 equivalence: |tax change| within {SESOI_TAX_POINTS} points ({construct})",
            sesoi=SESOI_TAX_POINTS / 100.0,
        )

    # The scale chain (S10, on more seeds at its endpoints)
    for level in ("L2b_shared_seq", "L3_per_task", "L4_oracle"):
        values = []
        for seed in seeds:
            small = _acc(index, "dispersed", level, seed, num_tasks=SCALE_T[0])
            large = _acc(index, "dispersed", level, seed, num_tasks=SCALE_T[-1])
            if small is not None and large is not None:
                values.append(large - small)
        out["secondary"][f"scale_{level}"] = paired_stats(
            values, f"scale: {level} at T={SCALE_T[-1]} - T={SCALE_T[0]} (dispersed)"
        )
    values = []
    for seed in seeds:
        small3 = _acc(index, "dispersed", "L3_per_task", seed, num_tasks=SCALE_T[0])
        small4 = _acc(index, "dispersed", "L4_oracle", seed, num_tasks=SCALE_T[0])
        large3 = _acc(index, "dispersed", "L3_per_task", seed, num_tasks=SCALE_T[-1])
        large4 = _acc(index, "dispersed", "L4_oracle", seed, num_tasks=SCALE_T[-1])
        if None not in (small3, small4, large3, large4):
            values.append((large4 - large3) - (small4 - small3))
    out["secondary"]["scale_tax_growth"] = paired_stats(
        values, f"scale: tax growth from T={SCALE_T[0]} to {SCALE_T[-1]} (dispersed)"
    )

    family = {
        name: stats.get("permutation_p") for name, stats in out["primary"].items()
    }
    out["holm"] = holm({k: v for k, v in family.items() if v is not None})
    out["westfall_young"] = westfall_young(
        {
            name: stats["per_seed"]
            for name, stats in out["primary"].items()
            if stats.get("n") == len(seeds)
        },
        len(seeds),
    )
    return out


# ---------------------------------------------------------------------------
# guards and report
# ---------------------------------------------------------------------------


def verify_against_anchors(cells: list[dict]) -> dict:
    """The operating point must reproduce S8; the scale endpoints S10."""
    out: dict = {}
    s8_path = "results/s8/s8_budget_study.json"
    if os.path.exists(s8_path):
        reference = {
            (c["construct"], c["level"]): c["accuracy"]
            for c in json.load(open(s8_path))["cells"]
            if c["rank"] == OPERATING["rank"]
            and c["protos"] == OPERATING["protos"]
            and c["top_k"] == OPERATING["top_k"]
            and c["seed"] == 42
        }
        index = _cells_index(cells)
        deltas = {}
        for construct in CONSTRUCTS:
            for level in (
                "L0_ncm",
                "L1_ridge",
                "L2b_shared_seq",
                "L3_per_task",
                "L4_oracle",
            ):
                cell = index.get((construct, level, 8, 1, 20, 42))
                if cell and (construct, level) in reference:
                    deltas[f"{construct}/{level}"] = abs(
                        reference[(construct, level)] - cell["accuracy"]
                    )
        out["s8_operating_point"] = {
            "checked": len(deltas),
            "max_abs_delta": max(deltas.values()) if deltas else None,
        }
    s10_path = "results/s10/s10_scaling_study.json"
    if os.path.exists(s10_path):
        reference = {
            (c["construct"], c["level"], c["num_tasks"], c["seed"]): c["accuracy"]
            for c in json.load(open(s10_path))["cells"]
            if c["dataset"] == "cifar100"
        }
        index = _cells_index(cells)
        deltas = {}
        for level in ("L2b_shared_seq", "L3_per_task", "L4_oracle"):
            for num_tasks in SCALE_T:
                for seed in (42, 1):
                    cell = index.get(("dispersed", level, 8, 1, num_tasks, seed))
                    key = ("dispersed", level, num_tasks, seed)
                    if cell and key in reference:
                        deltas[f"{level}/T{num_tasks}/seed{seed}"] = abs(
                            reference[key] - cell["accuracy"]
                        )
        out["s10_scale_endpoints"] = {
            "checked": len(deltas),
            "max_abs_delta": max(deltas.values()) if deltas else None,
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
            "command": "experiments/s11_confirmatory.py",
            "part": cell["part"],
            "construct": cell["construct"],
            "rank": cell["rank"],
            "prototypes_per_class": cell["protos"],
            "retention": cell["retention"],
            "routing": cell["routing"],
            "resources": resources,
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


def _print(report: dict) -> None:
    def fmt(stats: dict, scale: float = 100.0) -> str:
        if not stats or stats.get("n", 0) == 0:
            return "n/a"
        per = " ".join(f"{v * scale:+6.2f}" for v in stats["per_seed"])
        return (
            f"{per}   mean {stats['mean'] * scale:+6.2f} "
            f"sd {stats['sd'] * scale:5.2f} "
            f"CI [{stats['ci95'][0] * scale:+6.2f},{stats['ci95'][1] * scale:+6.2f}]"
        )

    print("\n" + "=" * 112)
    print("S11 CONFIRMATORY - paired differences per seed, exact small-sample tests")
    print("=" * 112)
    print("\nPRIMARY (Holm-corrected over the family)")
    for name, stats in report["primary"].items():
        holm_row = report["holm"].get(name, {})
        print(f"\n  {name}")
        print(f"    {stats['label']}")
        print(f"    per seed: {fmt(stats)}")
        wy = report.get("westfall_young", {}).get(name, {})
        print(
            f"    exact sign p={stats['sign_p']}  permutation p="
            f"{stats['permutation_p']:.4f}"
        )
        print(
            f"    WY max-statistic adjusted p={wy.get('adjusted_p', float('nan')):.4f}  "
            f"{'REJECT H0 at 0.05' if wy.get('adjusted_p', 1.0) < 0.05 else 'not rejected'}   "
            f"(Holm p={holm_row.get('adjusted_p', float('nan')):.4f}; infeasible at N=6)"
        )
    print("\nSECONDARY (exploratory, uncorrected)")
    for name, stats in report["secondary"].items():
        print(f"\n  {name}")
        print(f"    {stats['label']}")
        print(f"    per seed: {fmt(stats)}")
        line = (
            f"    exact sign p={stats['sign_p']}  permutation p="
            f"{stats['permutation_p']:.4f}"
        )
        if "within_sesoi" in stats:
            line += (
                f"  |mean|+CI <= {stats['sesoi'] * 100:.1f} points: "
                f"{stats['within_sesoi']}"
            )
        print(line)


def _print_order(payload: dict) -> None:
    order = payload.get("order_variance", {})
    if not order:
        return
    tax = order.get("tax_across_orders", {})
    l0 = order.get("levels", {}).get("L0_ncm", {})
    print(
        f"\n[S11 order variance] tax across nine order configurations: "
        f"mean {tax.get('mean', float('nan')) * 100:+.2f} "
        f"sd {tax.get('sd', float('nan')) * 100:.2f} "
        f"range [{tax.get('min', float('nan')) * 100:+.2f}, "
        f"{tax.get('max', float('nan')) * 100:+.2f}]"
    )
    print(
        f"  baseline for that spread: L0_ncm sd {l0.get('sd', float('nan')) * 100:.2f} "
        f"(S6 holds a task out, so even the order-invariant levels move); "
        f"the tax is the *more* stable quantity"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,1,2,3,4,5")
    parser.add_argument("--levels", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/s11")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="refresh hypotheses, order variance, contracts and anchors from the "
        "stored cells without training anything (used after editing a metric)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    study_path = os.path.join(args.out, "s11_confirmatory_study.json")

    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_func": args.lambda_func,
        "operating": OPERATING,
        "ranks": RANKS,
        "prototypes": PROTOTYPES,
        "scale_t": SCALE_T,
        "sesoi_tax_points": SESOI_TAX_POINTS,
        "constructs": CONSTRUCTS,
    }
    cells: list[dict] = []
    done: set[tuple] = set()
    if os.path.exists(study_path) and not args.force:
        previous = json.load(open(study_path))
        if previous.get("recipe") == recipe:
            cells = previous.get("cells", [])
            done = {_key(c) for c in cells}
            print(f"[S11] resuming: {len(done)} cells recorded", flush=True)

    def save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "s11_confirmatory",
            "backbone": "vit_b_16+proj768",
            "recipe": recipe,
            "seeds": seeds,
            "cells": cells,
            "contracts": {
                "__".join(str(part) for part in _key(c)): _contract(c) for c in cells
            },
            "hypotheses": build_hypotheses(cells, seeds) if cells else {},
            "order_variance": order_variance(),
            "anchors": verify_against_anchors(cells),
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)

    if args.report_only:
        if not os.path.exists(study_path):
            raise SystemExit(f"no study at {study_path}")
        payload = json.load(open(study_path))
        payload["hypotheses"] = build_hypotheses(payload["cells"], seeds)
        payload["order_variance"] = order_variance()
        payload["anchors"] = verify_against_anchors(payload["cells"])
        payload["contracts"] = {
            "__".join(str(part) for part in _key(c)): _contract(c)
            for c in payload["cells"]
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"[S11] refreshed {study_path} ({len(payload['cells'])} cells)")
        _print(payload["hypotheses"])
        _print_order(payload)
        print(f"\n[S11 anchors] {json.dumps(payload['anchors'], indent=1)}")
        return

    grid = default_grid(CONSTRUCTS, seeds, args.levels)
    print(
        f"[S11] grid: {len(grid)} cells, {len(done)} already done, seeds={seeds}",
        flush=True,
    )

    source_cache: dict = {}
    for cell in grid:
        if _key(cell) in done:
            continue
        result = run_cell(cell, args, device, source_cache)
        cells.append(result)
        done.add(_key(cell))
        save()
        print(
            f"[S11] {cell['part']} {cell['construct']:10s} {cell['level']:16s} "
            f"r={cell['rank']:<4d} p={cell['protos']:<3d} T={cell['num_tasks']:<3d} "
            f"seed={cell['seed']:<3d} acc={result['accuracy'] * 100:6.2f}%",
            flush=True,
        )

    save()
    print(f"[S11] wrote {study_path}", flush=True)
    payload = json.load(open(study_path))
    _print(payload["hypotheses"])
    print(f"\n[S11 anchors] {json.dumps(payload['anchors'], indent=1)}")
    _print_order(payload)


if __name__ == "__main__":
    main()
