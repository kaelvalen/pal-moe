"""
Router Ranking Study - R2, the pre-registered hypothesis.

    docs/ROUTER_RANKING_PREREG.md

Question: can better ranking realize the existing expert capacity?

    R0  prototype ranking, one mean per class   measured at six seeds (S8/S11)
    R1  prototype ranking, 16 per class         measured at six seeds (S8/S11)
    R2  a `z -> T` supervised task-compatibility scorer, trained against the
        observed task identity                <- this script

R2 is the v1 `DynamicRouter` gate: a single `nn.Linear(dim, num_experts)` whose
target is the *task identity* observed during training, with old expert rows
frozen as tasks arrive. At inference no task id is available - the gate sees
only `z` and produces `T` routing logits.

**Why the bank is provably untouched.** The ladder trains its expert and readout
with the *task index* as the routing target (`fit_task` passes `ids = task_index`,
never the router's output), so the expert bank's training trajectory does not
depend on the router at all. R2's gate is therefore fitted *post hoc* on the
cached features, after the bank is trained, and the bank is literally the same
object in both arms. The router also cannot receive gradients from the experts:
the gate's loss is a function of `gate(z)` alone.

The primary endpoint is coverage, not accuracy:

    Delta C@3 = C_R2@3 - C_R0@3

with the routing ground truth being the **owner expert** (the task assignment),
not `L4`'s prediction. `L4` is the classification oracle and is router
independent, so it must be identical in both arms - that is the guard.

Usage:
    python experiments/rr_ranking.py --device cuda
    python experiments/rr_ranking.py --seeds 42 --report-only
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import s2_ladder  # noqa: E402
import s6b_difficulty  # noqa: E402
import s10_scaling  # noqa: E402
import s11_confirmatory as s11  # noqa: E402
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402
from pal_moe.models.router import DynamicRouter  # noqa: E402

SOURCE_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"
REGIMES = ["coherent", "dispersed"]
OPERATING = {"rank": 8, "protos": 1, "top_k": 1, "num_tasks": 20}
# The gate's declared budget: the *same* as the expert's (10 epochs, batch 128,
# lr 1e-3), so no budget asymmetry can explain a difference between R0 and R2.
GATE_EPOCHS = 10
GATE_BATCH = 128
GATE_LR = 1e-3
# Exploratory sensitivity: budgets far above the declared one, to test whether
# "R2 loses" is a tuning artefact rather than a structural one.
GATE_BUDGETS = [(10, 1e-3), (50, 1e-3), (200, 1e-2)]
CANDIDATE_SIZES = [1, 2, 3, 4, 8]


# ---------------------------------------------------------------------------
# the R2 router: a learned z -> T compatibility scorer
# ---------------------------------------------------------------------------


class LearnedGateRouter(torch.nn.Module):
    """v1's `DynamicRouter` gate, wrapped in the S1 router interface.

    Implements exactly what `LadderModel` asks of a router - `top_k` and
    `register_class` - so the ladder code path is unchanged. `register_class` is
    a no-op: the gate does not store prototypes, it scores `z` directly. The gate
    itself is v1's, including its `lock_historical_routing` freeze hooks.
    """

    def __init__(self, router: DynamicRouter, num_experts: int):
        super().__init__()
        self.router = router
        self.num_experts = int(num_experts)
        self.temperature = float(router.temperature)
        # The ladder reads `counts` only for prototype routers; an empty buffer
        # keeps `resources()` honest about what this router stores.
        self.register_buffer("counts", torch.zeros(0))

    @torch.no_grad()
    def register_class(self, class_id: int, expert_id: int, z: torch.Tensor) -> None:
        return None

    def expert_scores(self, z: torch.Tensor) -> torch.Tensor:
        # L2-normalised input, matching the scale the prototype ranking works at
        # (cosine similarity is scale-free), so the gate is not handed a feature
        # magnitude the prototype scorer never used.
        return self.router.gate(torch.nn.functional.normalize(z.float(), dim=-1))

    def route(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.expert_scores(z), dim=-1)

    def top_k(self, z: torch.Tensor, k: int = 1):
        scores = self.expert_scores(z)
        values, indices = scores.topk(min(k, self.num_experts), dim=-1)
        return indices, values

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.route(z)


def train_gate(
    router: DynamicRouter,
    tasks: list[dict],
    device,
    epochs: int,
    lr: float,
    batch_size: int = GATE_BATCH,
    seed: int = 0,
) -> dict:
    """Fit the gate incrementally, using v1's own freeze policy.

    `lock_historical_routing(t)` zeroes the gradient of the first `t` expert rows
    (the hooks `DynamicRouter` installs in `__init__`), so when task `t` arrives
    only its own row is optimised and an old expert's scorer cannot be rewritten.
    The target is the observed task identity; the input is the current task's
    features. No task id is used at inference. The budget matches the expert's
    (mini-batches of 128, ten epochs, lr 1e-3).

    The gate receives no gradient from the experts: its loss is a function of
    `gate(z)` alone, and the bank is trained before the gate is fitted.
    """
    router.train()
    losses = []
    for t, task in enumerate(tasks):
        router.lock_historical_routing(t)
        feats = torch.nn.functional.normalize(
            task["splits"]["train"][0].to(device), dim=-1
        )
        targets = torch.full((feats.size(0),), t, dtype=torch.long, device=device)
        optimizer = torch.optim.Adam(router.parameters(), lr=lr)
        seen = torch.arange(t + 1, device=device)
        generator = torch.Generator().manual_seed(seed + t)
        loss = None
        for _ in range(epochs):
            perm = torch.randperm(feats.size(0), generator=generator)
            for start in range(0, feats.size(0) - batch_size + 1, batch_size):
                index = perm[start : start + batch_size]
                _, _, raw_logits = router(feats[index])
                loss = torch.nn.functional.cross_entropy(
                    raw_logits[:, seen], targets[index]
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach()))
    router.eval()
    return {"final_loss": losses[-1], "losses": losses, "epochs": epochs, "lr": lr}


def fit_gate(tasks, dim, device, epochs: int, lr: float, seed: int) -> DynamicRouter:
    router = DynamicRouter(
        input_dim=dim,
        num_experts=len(tasks),
        top_k=OPERATING["top_k"],
        temperature=1.0,
    ).to(device)
    train_gate(router, tasks, device, epochs, lr, seed=seed)
    return router


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def bank_hash(model) -> str:
    """A fingerprint of everything the routing must not touch."""
    digest = hashlib.sha256()
    tensors = list(model.readout.parameters()) + [
        p for expert in model.experts for p in expert.parameters()
    ]
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().float().numpy().tobytes())
    return digest.hexdigest()[:16]


def train_bank(regime: str, seed: int, args, device):
    """The S11 `L3_per_task` training, unchanged: the frozen expert bank."""
    _, source = s2_ladder.load_tasks(SOURCE_CACHE)
    tasks = s6b_difficulty.build_construction(source, regime)
    cell = {
        "construct": regime,
        "level": "L3_per_task",
        "rank": OPERATING["rank"],
        "protos": OPERATING["protos"],
        "top_k": OPERATING["top_k"],
        "num_tasks": OPERATING["num_tasks"],
        "seed": seed,
    }
    model = s11.train_model("L3_per_task", tasks, cell, args, device)
    return model, tasks


def evaluate_arm(model, tasks: list[dict]) -> dict:
    """Accuracy plus the candidate-set decomposition, same definitions as S10."""
    accuracy = s11.evaluate(model, "L3_per_task", tasks)
    routing = s10_scaling.routing_report(model, tasks)
    coverage = routing.get("coverage", {})
    conditional_oracle = routing.get("conditional_oracle", {})
    ceiling = {
        m: coverage[m] * conditional_oracle[m]
        for m in coverage
        if m in conditional_oracle
    }
    return {
        "accuracy": accuracy["accuracy"],
        "acc_matrix": accuracy["acc_matrix"],
        "routing": routing,
        "ceiling": ceiling,
    }


def run_cell(regime: str, seed: int, args, device) -> dict:
    args.seed = seed
    model, tasks = train_bank(regime, seed, args, device)
    fingerprint = bank_hash(model)

    # R0: the prototype router the bank was built with.
    r0 = evaluate_arm(model, tasks)

    # The oracle route does not use the router, so its accuracy must be
    # bit-identical before and after the swap. This is the guard that the
    # manipulation is ranking-only.
    def oracle_accuracy():
        values = [
            model.evaluate_task(task, oracle=True, task_id=i)
            for i, task in enumerate(tasks)
        ]
        return float(np.mean(values))

    oracle_before = oracle_accuracy()

    # R2: swap in the learned gate, fitted post hoc on the frozen cache.
    dim = int(tasks[0]["splits"]["train"][0].size(1))
    gate_router = fit_gate(
        tasks, dim, device, args.gate_epochs, args.gate_lr, seed=seed
    )
    model.router = LearnedGateRouter(gate_router, len(tasks)).to(device)
    r2 = evaluate_arm(model, tasks)
    oracle_after = oracle_accuracy()

    # Exploratory: does a much larger gate budget change the verdict?
    sensitivity = {}
    for epochs, lr in GATE_BUDGETS:
        router = fit_gate(tasks, dim, device, epochs, lr, seed=seed)
        model.router = LearnedGateRouter(router, len(tasks)).to(device)
        arm = evaluate_arm(model, tasks)
        sensitivity[f"{epochs}ep_lr{lr:g}"] = {
            "accuracy": arm["accuracy"],
            "coverage_at_3": arm["routing"]["coverage"].get("3"),
        }
    model.router = LearnedGateRouter(gate_router, len(tasks)).to(device)

    num_classes = sum(len(t["classes"]) for t in tasks)
    return {
        "regime": regime,
        "seed": seed,
        "bank_hash": fingerprint,
        "bank_hash_after_r2": bank_hash(model),
        "l4_invariance": abs(oracle_before - oracle_after),
        "l4_accuracy": oracle_after,
        "gate_train": {"epochs": args.gate_epochs, "lr": args.gate_lr},
        "gate_sensitivity": sensitivity,
        "R0": r0,
        "R2": r2,
        "num_tasks": len(tasks),
        "num_classes": num_classes,
        "resources": {
            "R0_router_bytes": num_classes * dim * 4,
            "R2_router_bytes": (len(tasks) * (dim + 1)) * 4,
        },
    }


# ---------------------------------------------------------------------------
# hypotheses
# ---------------------------------------------------------------------------


def build_hypotheses(cells: list[dict], seeds: list[int]) -> dict:
    """The pre-registered endpoints, paired over seeds, plus the guards."""

    def per_seed(values):
        return [v for v in values if v is not None]

    delta_c3 = {}
    delta_tax = {}
    delta_ratio = {}
    for regime in REGIMES:
        rows = {c["seed"]: c for c in cells if c["regime"] == regime}
        delta_c3[regime] = per_seed(
            [
                rows[s]["R2"]["routing"]["coverage"].get("3")
                - rows[s]["R0"]["routing"]["coverage"].get("3")
                for s in seeds
                if s in rows
            ]
        )
        # L4 is router independent: read it from S11 at the same seeds.
        l4 = s11_lookup(seeds, regime, "L4_oracle")
        delta_tax[regime] = per_seed(
            [
                (l4[s] - rows[s]["R2"]["accuracy"])
                - (l4[s] - rows[s]["R0"]["accuracy"])
                for s in seeds
                if s in rows and s in l4
            ]
        )
        l0 = s11_lookup(seeds, regime, "L0_ncm")
        values = []
        for s in seeds:
            if s not in rows or s not in l4 or s not in l0:
                continue
            denominator = l4[s] - l0[s]
            if denominator <= 1e-9:
                continue
            values.append(
                (rows[s]["R2"]["accuracy"] - l0[s]) / denominator
                - (rows[s]["R0"]["accuracy"] - l0[s]) / denominator
            )
        delta_ratio[regime] = values

    out = {
        "primary": {
            f"delta_C3_{regime}": s11.paired_stats(
                delta_c3[regime], f"Delta C@3 (R2 - R0, {regime})"
            )
            for regime in REGIMES
        },
        "secondary": {
            f"delta_tax_{regime}": s11.paired_stats(
                delta_tax[regime], f"Delta tax (R2 - R0, {regime})"
            )
            for regime in REGIMES
        },
        "headroom_fraction": {},
    }
    for regime in REGIMES:
        out["secondary"][f"delta_R_iso_ncm_{regime}"] = s11.paired_stats(
            delta_ratio[regime], f"Delta R_iso_ncm (R2 - R0, {regime})"
        )
        # Descriptive only: the fraction of R0's m=3 headroom recovered.
        # Headroom is an *accuracy* difference: what a perfect selector over the
        # router's candidates would add over the learned top-1 route, on the
        # covered set. (coverage * conditional) for both terms, so the fraction
        # is comparable across arms even when coverage moves.
        rows = [c for c in cells if c["regime"] == regime]
        fractions = []
        for cell in rows:
            cov0 = cell["R0"]["routing"]["coverage"].get("3")
            cov2 = cell["R2"]["routing"]["coverage"].get("3")
            learned0 = cell["R0"]["routing"]["conditional_learned"].get("3")
            learned2 = cell["R2"]["routing"]["conditional_learned"].get("3")
            oracle0 = cell["R0"]["routing"]["conditional_oracle"].get("3")
            if None in (cov0, cov2, learned0, learned2, oracle0):
                continue
            headroom0 = cov0 * oracle0 - cov0 * learned0
            if headroom0 <= 1e-9:
                continue
            fractions.append((cov2 * learned2 - cov0 * learned0) / headroom0)
        out["headroom_fraction"][regime] = s11.paired_stats(
            fractions,
            f"headroom fraction recovered at m=3 ({regime})",
        )
    family = {
        name: stats.get("permutation_p")
        for name, stats in {**out["primary"], **out["secondary"]}.items()
    }
    out["westfall_young"] = s11.westfall_young(
        {
            name: stats["per_seed"]
            for name, stats in {**out["primary"], **out["secondary"]}.items()
            if stats.get("n") == len(seeds)
        },
        len(seeds),
    )
    out["holm_reference"] = s11.holm({k: v for k, v in family.items() if v is not None})
    return out


_S11_CACHE: dict = {}


def s11_lookup(seeds: list[int], regime: str, level: str) -> dict:
    """Router-independent levels, read from S11 at the same seeds."""
    key = (tuple(seeds), regime, level)
    if key not in _S11_CACHE:
        path = "results/s11/s11_confirmatory_study.json"
        cells = json.load(open(path))["cells"]
        _S11_CACHE[key] = {
            c["seed"]: c["accuracy"]
            for c in cells
            if c["construct"] == regime
            and c["level"] == level
            and c["rank"] == OPERATING["rank"]
            and c["protos"] == OPERATING["protos"]
            and c["num_tasks"] == OPERATING["num_tasks"]
        }
    return _S11_CACHE[key]


def guards(cells: list[dict], seeds: list[int]) -> dict:
    """R0 must reproduce S11; L4 must be untouched; the bank must not move."""
    r0_deltas = {}
    l4_deltas = {}
    bank_moved = 0
    for cell in cells:
        regime, seed = cell["regime"], cell["seed"]
        reference = s11_lookup(seeds, regime, "L3_per_task").get(seed)
        if reference is not None:
            r0_deltas[f"{regime}/{seed}"] = abs(reference - cell["R0"]["accuracy"])
        l4_deltas[f"{regime}/{seed}"] = cell["l4_invariance"]
        if cell["bank_hash"] != cell["bank_hash_after_r2"]:
            bank_moved += 1
    return {
        "r0_reproduces_s11": {
            "checked": len(r0_deltas),
            "max_abs_delta": max(r0_deltas.values()) if r0_deltas else None,
        },
        "l4_invariance_across_router_swap": {
            "checked": len(l4_deltas),
            "max_abs_delta": max(l4_deltas.values()) if l4_deltas else None,
        },
        "bank_hashes_moved_by_r2": bank_moved,
    }


def _contract(cell: dict) -> dict:
    record = build_run_record(
        factors={
            "dataset": "cifar100",
            "protocol": "class_il",
            "task_id_at_inference": False,
            "class_masking": False,
            "routing_mode": "learned",
            "increment_type": "class",
            "class_space": "task",
            "num_tasks": cell["num_tasks"],
            "classes_per_task": 5,
            "seed": cell["seed"],
            "task_order_seed": None,
            "model_family": "L3_per_task+R2_gate",
            "backbone": "vit_b_16+proj768",
            "backbone_pretraining": "imagenet_frozen",
            "readout": "cosine",
            "expert": "residual_adapter",
        },
        metrics={
            "learning": {
                "accuracy": cell["R2"]["accuracy"],
                "forgetting": 0.0,
                "acc_matrix": cell["R2"]["acc_matrix"],
            },
            "cost": {
                "stored_bytes": cell["resources"]["R2_router_bytes"],
                "total_params": 0,
            },
        },
        provenance={
            "command": "experiments/rr_ranking.py",
            "regime": cell["regime"],
            "R0_accuracy": cell["R0"]["accuracy"],
            "R2_accuracy": cell["R2"]["accuracy"],
            "bank_hash": cell["bank_hash"],
            "coverage_R0": cell["R0"]["routing"].get("coverage"),
            "coverage_R2": cell["R2"]["routing"].get("coverage"),
            "conditional_oracle_R0": cell["R0"]["routing"].get("conditional_oracle"),
            "conditional_oracle_R2": cell["R2"]["routing"].get("conditional_oracle"),
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


def _print(report: dict) -> None:
    def fmt(stats, scale=1.0):
        if not stats or stats.get("n", 0) == 0:
            return "n/a"
        per = " ".join(f"{v * scale:+.4f}" for v in stats["per_seed"])
        return (
            f"{per}   mean {stats['mean'] * scale:+.4f} "
            f"sd {stats['sd'] * scale:.4f} "
            f"CI [{stats['ci95'][0] * scale:+.4f},{stats['ci95'][1] * scale:+.4f}]"
        )

    print("\n" + "=" * 104)
    print("ROUTER RANKING STUDY - R2 vs R0")
    print("=" * 104)
    for family in ("primary", "secondary"):
        print(f"\n{family.upper()}")
        for name, stats in report[family].items():
            wy = report.get("westfall_young", {}).get(name, {})
            print(f"\n  {name}")
            print(f"    {stats['label']}")
            print(f"    per seed: {fmt(stats)}")
            print(
                f"    exact sign p={stats['sign_p']}  permutation p="
                f"{stats['permutation_p']:.4f}  WY adjusted p="
                f"{wy.get('adjusted_p', float('nan')):.4f}  "
                f"{'REJECT' if wy.get('adjusted_p', 1.0) < 0.05 else 'not rejected'}"
            )
    print("\nHEADROOM FRACTION RECOVERED (descriptive, not a success threshold)")
    for regime, stats in report.get("headroom_fraction", {}).items():
        print(f"  {regime}: {fmt(stats)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,1,2,3,4,5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument("--gate_epochs", type=int, default=GATE_EPOCHS)
    parser.add_argument("--gate_lr", type=float, default=GATE_LR)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/rr")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    study_path = os.path.join(args.out, "rr_ranking_study.json")

    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_func": args.lambda_func,
        "gate_epochs": args.gate_epochs,
        "gate_lr": args.gate_lr,
        "operating": OPERATING,
        "regimes": REGIMES,
        "candidate_sizes": CANDIDATE_SIZES,
    }
    cells: list[dict] = []
    done: set[tuple] = set()
    if os.path.exists(study_path) and not args.force:
        previous = json.load(open(study_path))
        if previous.get("recipe") == recipe:
            cells = previous.get("cells", [])
            done = {(c["regime"], c["seed"]) for c in cells}
            print(f"[RR] resuming: {len(done)} cells recorded", flush=True)

    def save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "rr_ranking",
            "prereg": "docs/ROUTER_RANKING_PREREG.md",
            "backbone": "vit_b_16+proj768",
            "recipe": recipe,
            "seeds": seeds,
            "cells": cells,
            "contracts": {
                f"{c['regime']}__seed{c['seed']}": _contract(c) for c in cells
            },
            "hypotheses": build_hypotheses(cells, seeds) if cells else {},
            "guards": guards(cells, seeds) if cells else {},
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)

    if args.report_only:
        payload = json.load(open(study_path))
        payload["hypotheses"] = build_hypotheses(payload["cells"], seeds)
        payload["guards"] = guards(payload["cells"], seeds)
        payload["contracts"] = {
            f"{c['regime']}__seed{c['seed']}": _contract(c) for c in payload["cells"]
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"[RR] refreshed {study_path} ({len(payload['cells'])} cells)")
        _print(payload["hypotheses"])
        print(f"\n[RR guards] {json.dumps(payload['guards'], indent=1)}")
        return

    grid = [
        {"regime": regime, "seed": seed}
        for regime in REGIMES
        for seed in seeds
        if (regime, seed) not in done
    ]
    print(
        f"[RR] grid: {len(grid)} new cells (of {len(REGIMES) * len(seeds)}), "
        f"seeds={seeds}",
        flush=True,
    )
    for cell in grid:
        result = run_cell(cell["regime"], cell["seed"], args, device)
        cells.append(result)
        save()
        print(
            f"[RR] {cell['regime']:10s} seed={cell['seed']:<3d} "
            f"R0 acc={result['R0']['accuracy'] * 100:6.2f} C@3="
            f"{(result['R0']['routing']['coverage'].get('3') or float('nan')):.4f}  "
            f"R2 acc={result['R2']['accuracy'] * 100:6.2f} C@3="
            f"{(result['R2']['routing']['coverage'].get('3') or float('nan')):.4f}  "
            f"gate budget={result['gate_train']['epochs']}ep/lr"
            f"{result['gate_train']['lr']:g}  sensitivity C@3="
            + ",".join(
                f"{(v['coverage_at_3'] or 0):.3f}"
                for v in result["gate_sensitivity"].values()
            ),
            flush=True,
        )

    save()
    print(f"[RR] wrote {study_path}", flush=True)
    payload = json.load(open(study_path))
    _print(payload["hypotheses"])
    print(f"\n[RR guards] {json.dumps(payload['guards'], indent=1)}")


if __name__ == "__main__":
    main()
