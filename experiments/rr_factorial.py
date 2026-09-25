"""
Representation x Routing Objective - the four-arm factorial.

    docs/REPRESENTATION_ROUTING_PREREG.md

    arm              representation            routing objective
    frozen_off       raw z                     prototype ranking (Stage 1's ladder)
    frozen_on        raw z                     globally consistent supervised gate
    trainable_off    candidate-specific E_e(z) prototype ranking in the adapted space
    trainable_on     candidate-specific E_e(z) prototype ranking + routing loss in the
                                               expert's training loop

Two pinned properties the arms depend on:

**Prototype timing is unambiguous here.** The ladder's freeze policy trains only
the newest expert, so `E_t` stops changing after task `t` and the task's
prototypes are the same whether they are taken immediately after task `t` or at
the end of the run. The stored evidence is therefore task-local sufficient
statistics (snapshot semantics) in every arm.

**The routing objective is globally consistent and rehearsal-free.** It is
trained on the stored prototypes of *all* seen experts with cross-entropy over
all of them, so no row is ever trained one-vs-previous - the structural failure
that sank R2 in the Router Ranking Study. In the interaction arm the loss enters
the expert's training loop, using the stored prototypes of the older experts and
a stop-gradient EMA anchor for the current expert; expert gradients flow only
through the current encoded sample, never through prototype construction.

The primary endpoint is `Delta C@3` with the routing ground truth being the owner
expert, not `L4`.

Usage:
    python experiments/rr_factorial.py --device cuda
    python experiments/rr_factorial.py --seeds 42 --arms frozen_off trainable_off
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import s10_scaling  # noqa: E402
import s11_confirmatory as s11  # noqa: E402
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402

SOURCE_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"
REGIMES = ["coherent", "dispersed"]
ARMS = ["frozen_off", "frozen_on", "trainable_off", "trainable_on"]
OPERATING = {"rank": 8, "protos": 1, "top_k": 1, "num_tasks": 20}
ROUTING_LAMBDA = 1.0  # matches the ladder's existing prototype-anchor weight
ANCHOR_EMA = 0.9


# ---------------------------------------------------------------------------
# routers
# ---------------------------------------------------------------------------


def adapted_scores(experts, prototypes: dict, z: torch.Tensor) -> torch.Tensor:
    """`max_c cos(E_e(z), p_{e,c})` for every expert that has prototypes."""
    out = torch.full((z.size(0), len(experts)), -2.0, device=z.device)
    for expert_id, prototype in prototypes.items():
        if expert_id >= len(experts):
            continue
        h = F.normalize(experts[expert_id].transform(z), dim=-1)
        p = F.normalize(prototype.to(z.device), dim=-1)
        out[:, expert_id] = (h @ p.t()).max(dim=-1).values
    return out


class AdaptedSpaceRouter(torch.nn.Module):
    """Candidate-specific ranking: each expert scores in its own adapted space."""

    def __init__(self, experts, num_experts: int, dim: int):
        super().__init__()
        self.experts = experts
        self.num_experts = int(num_experts)
        self.dim = int(dim)
        self.temperature = 1.0
        self.prototypes: dict[int, torch.Tensor] = {}
        self.register_buffer("counts", torch.zeros(0))

    @torch.no_grad()
    def register_class(self, class_id: int, expert_id: int, z: torch.Tensor) -> None:
        return None

    @torch.no_grad()
    def set_prototypes(self, expert_id: int, prototype: torch.Tensor) -> None:
        self.prototypes[int(expert_id)] = prototype.detach().float().cpu()

    def expert_scores(self, z: torch.Tensor) -> torch.Tensor:
        return adapted_scores(self.experts, self.prototypes, z)

    def route(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.expert_scores(z), dim=-1)

    def top_k(self, z: torch.Tensor, k: int = 1):
        scores = self.expert_scores(z)
        values, indices = scores.topk(min(k, self.num_experts), dim=-1)
        return indices, values

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.route(z)


class GateRouter(torch.nn.Module):
    """A gate trained on the stored prototypes, with the S1 router interface."""

    def __init__(self, gate: torch.nn.Linear, num_experts: int):
        super().__init__()
        self.gate = gate
        self.num_experts = int(num_experts)
        self.temperature = 1.0
        self.register_buffer("counts", torch.zeros(0))

    @torch.no_grad()
    def register_class(self, class_id: int, expert_id: int, z: torch.Tensor) -> None:
        return None

    def expert_scores(self, z: torch.Tensor) -> torch.Tensor:
        return self.gate(F.normalize(z.float(), dim=-1))

    def route(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.expert_scores(z), dim=-1)

    def top_k(self, z: torch.Tensor, k: int = 1):
        scores = self.expert_scores(z)
        values, indices = scores.topk(min(k, self.num_experts), dim=-1)
        return indices, values

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.route(z)


def train_gate_on_prototypes(
    model, num_experts: int, epochs: int = 500, lr: float = 1e-2
) -> GateRouter:
    """Fit `z -> T` on the stored class prototypes, globally and rehearsal-free.

    Every prototype of every seen expert is a training example with its owner
    expert as the target, and the cross-entropy is over *all* experts on every
    step. No row is trained one-vs-previous, which is what failed in the Router
    Ranking Study.
    """
    dim = model.dim
    gate = torch.nn.Linear(dim, num_experts, bias=True).to(model.device)
    counts = model.router.counts
    seen = torch.nonzero(counts > 0).flatten()
    feats = model.router.means[seen].clone()
    owners = model.router.class_expert[seen].clone()
    feats = F.normalize(feats.float(), dim=-1)
    optimizer = torch.optim.Adam(gate.parameters(), lr=lr)
    for _ in range(epochs):
        loss = F.cross_entropy(gate(feats), owners)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    gate.eval()
    return GateRouter(gate, num_experts).to(model.device)


# ---------------------------------------------------------------------------
# the interaction arm's routing loss
# ---------------------------------------------------------------------------


class RoutingLoss:
    """The interaction arm's extra term, added to the expert's training loop.

    Older experts are scored against their stored prototypes (snapshot
    semantics). The current expert is scored against a stop-gradient EMA of its
    own adapted features, so no gradient flows through prototype construction.
    """

    def __init__(self, model, lam: float = ROUTING_LAMBDA, ema: float = ANCHOR_EMA):
        self.model = model
        self.lam = lam
        self.ema = ema
        self.anchor: dict[int, torch.Tensor] = {}

    def __call__(self, z: torch.Tensor, y: torch.Tensor, task_index) -> torch.Tensor:
        if task_index is None or not self.model.experts:
            return torch.zeros((), device=z.device)
        current = int(task_index)
        experts = self.model.experts

        prototypes: dict[int, torch.Tensor] = {}
        router = self.model.router
        if router is not None and router.counts.numel():
            seen = torch.nonzero(router.counts > 0).flatten()
            owners = router.class_expert[seen]
            means = router.means[seen]
            for expert_id in torch.unique(owners).tolist():
                prototypes[int(expert_id)] = means[owners == expert_id]

        scores = adapted_scores(experts, prototypes, z)
        # The current expert's own anchor: a stop-gradient EMA of E_current(z).
        with torch.no_grad():
            adapted = experts[current].transform(z).mean(dim=0)
            if current in self.anchor:
                self.anchor[current] = (
                    self.ema * self.anchor[current] + (1 - self.ema) * adapted
                )
            else:
                self.anchor[current] = adapted
        h = F.normalize(experts[current].transform(z), dim=-1)
        a = F.normalize(self.anchor[current].detach().to(z.device), dim=-1)
        scores[:, current] = h @ a
        targets = torch.full((z.size(0),), current, dtype=torch.long, device=z.device)
        return self.lam * F.cross_entropy(scores, targets)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def bank_hash(model) -> str:
    digest = hashlib.sha256()
    tensors = list(model.readout.parameters()) + [
        p for expert in model.experts for p in expert.parameters()
    ]
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().float().numpy().tobytes())
    return digest.hexdigest()[:16]


def prototype_hash(prototypes) -> str:
    digest = hashlib.sha256()
    for key in sorted(prototypes):
        digest.update(
            prototypes[key].detach().cpu().contiguous().float().numpy().tobytes()
        )
    return digest.hexdigest()[:16]


def train_bank(regime: str, seed: int, args, device, joint: bool):
    """The S11 `L3_per_task` training, plus the routing loss when `joint`."""
    _, source = s11.s2_ladder.load_tasks(SOURCE_CACHE)
    tasks = s11.s6b_difficulty.build_construction(source, regime)
    cell = {
        "construct": regime,
        "level": "L3_per_task",
        "rank": OPERATING["rank"],
        "protos": OPERATING["protos"],
        "top_k": OPERATING["top_k"],
        "num_tasks": OPERATING["num_tasks"],
        "seed": seed,
    }
    args.seed = seed
    model = s11.train_model("L3_per_task", tasks, cell, args, device)
    return model, tasks


def train_bank_joint(regime: str, seed: int, args, device, tasks=None):
    """Train the bank with the interaction arm's routing loss in the loop."""
    if tasks is None:
        _, source = s11.s2_ladder.load_tasks(SOURCE_CACHE)
        tasks = s11.s6b_difficulty.build_construction(source, regime)
    dim = int(tasks[0]["splits"]["train"][0].size(1))
    num_classes = sum(len(t["classes"]) for t in tasks)
    spec = s11.s2_ladder.LEVELS_BY_NAME["L3_per_task"]
    cell_args = argparse.Namespace(
        rank=OPERATING["rank"],
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=seed,
        lambda_func=args.lambda_func,
        max_experts=max(20, len(tasks)),
    )
    s11.s2_ladder.set_seed(seed)
    model = s11.s2_ladder.LadderModel(
        spec, dim, num_classes, cell_args, device, router_slots=num_classes
    )
    model.router_prototypes = OPERATING["protos"]
    model.top_k_experts = OPERATING["top_k"]
    model.extra_loss = RoutingLoss(model)
    for t, task in enumerate(tasks):
        model.seen = sorted(set(model.seen) | set(task["classes"]))
        model.fit_task(task, t)
        model.register_task(task, t)
    model.extra_loss = None
    return model, tasks


def register_adapted_prototypes(model, tasks) -> dict:
    """Task-local sufficient statistics in the adapted space, per expert."""
    prototypes = {}
    for expert_id, task in enumerate(tasks):
        feats, labels = task["splits"]["train"]
        feats, labels = feats.to(model.device), labels.to(model.device)
        with torch.no_grad():
            adapted = model.experts[expert_id].transform(feats)
        rows = []
        for c in task["classes"]:
            mask = labels == c
            if bool(mask.any()):
                rows.append(adapted[mask].mean(dim=0).detach().cpu())
        if rows:
            prototypes[expert_id] = torch.stack(rows)
    return prototypes


def evaluate_arm(model, tasks) -> dict:
    accuracy = s11.evaluate(model, "L3_per_task", tasks)
    routing = s10_scaling.routing_report(model, tasks)
    coverage = routing.get("coverage", {})
    conditional_oracle = routing.get("conditional_oracle", {})
    return {
        "accuracy": accuracy["accuracy"],
        "acc_matrix": accuracy["acc_matrix"],
        "routing": routing,
        "ceiling": {
            m: coverage[m] * conditional_oracle[m]
            for m in coverage
            if m in conditional_oracle
        },
    }


def run_cell(regime: str, seed: int, arm: str, args, device, cache: dict) -> dict:
    key = (regime, seed)
    if key not in cache:
        base_model, tasks = train_bank(regime, seed, args, device, joint=False)
        cache[key] = {"base_model": base_model, "tasks": tasks}
    entry = cache[key]
    tasks = entry["tasks"]
    dim = int(tasks[0]["splits"]["train"][0].size(1))

    # A copy per arm: the arms share the trained bank but differ in their
    # router, and mutating a cached model would make the result depend on the
    # arm order. The frozen arms keep the prototype router the ladder built.
    if arm == "trainable_on":
        joint_key = (regime, seed, "joint")
        if joint_key not in cache:
            joint_model, joint_tasks = train_bank_joint(
                regime, seed, args, device, tasks
            )
            cache[joint_key] = {"base_model": joint_model, "tasks": joint_tasks}
        model = copy.deepcopy(cache[joint_key]["base_model"])
    else:
        model = copy.deepcopy(entry["base_model"])

    if arm == "frozen_off":
        prototypes = None  # the prototype router the ladder trained with
    elif arm == "frozen_on":
        model.router = train_gate_on_prototypes(model, len(tasks))
        prototypes = None
    elif arm in ("trainable_off", "trainable_on"):
        prototypes = register_adapted_prototypes(model, tasks)
        router = AdaptedSpaceRouter(model.experts, len(tasks), dim).to(device)
        for expert_id, block in prototypes.items():
            router.set_prototypes(expert_id, block)
        model.router = router
    else:
        raise KeyError(arm)

    result = evaluate_arm(model, tasks)
    raw_prototypes = {}
    # The raw-space prototype hash is taken from the *bank's* registered means,
    # which live in the prototype router the ladder built, not in `model.router`
    # (which an arm may have replaced).
    ladder_router = entry["base_model"].router
    if ladder_router is not None and ladder_router.counts.numel():
        counts = ladder_router.counts
        raw_prototypes = {0: ladder_router.means[counts > 0].clone()}
    return {
        "regime": regime,
        "seed": seed,
        "arm": arm,
        "accuracy": result["accuracy"],
        "acc_matrix": result["acc_matrix"],
        "routing": result["routing"],
        "ceiling": result["ceiling"],
        "bank_hash": bank_hash(model),
        "prototype_hash": (
            prototype_hash(prototypes)
            if prototypes
            else (prototype_hash(raw_prototypes) if raw_prototypes else None)
        ),
        "num_tasks": len(tasks),
        "prototype_source": (
            "raw z"
            if arm in ("frozen_off", "frozen_on")
            else "E_e(z), snapshot after task e"
        ),
    }


# ---------------------------------------------------------------------------
# hypotheses
# ---------------------------------------------------------------------------


def _index(cells: list[dict]) -> dict:
    return {(c["regime"], c["arm"], c["seed"]): c for c in cells}


def build_hypotheses(cells: list[dict], seeds: list[int]) -> dict:
    index = _index(cells)

    def acc(regime, arm, seed, key="accuracy"):
        cell = index.get((regime, arm, seed))
        return cell[key] if cell else None

    def cov(regime, arm, seed, m="3"):
        cell = index.get((regime, arm, seed))
        if not cell:
            return None
        return cell["routing"].get("coverage", {}).get(m)

    effects = {}
    for regime in REGIMES:
        effects[f"routing_objective_{regime}"] = s11.paired_stats(
            [
                cov(regime, "frozen_on", s) - cov(regime, "frozen_off", s)
                for s in seeds
                if None
                not in (cov(regime, "frozen_on", s), cov(regime, "frozen_off", s))
            ],
            f"routing objective with frozen representation (Delta C@3, {regime})",
        )
        effects[f"expert_adapted_{regime}"] = s11.paired_stats(
            [
                cov(regime, "trainable_off", s) - cov(regime, "frozen_off", s)
                for s in seeds
                if None
                not in (
                    cov(regime, "trainable_off", s),
                    cov(regime, "frozen_off", s),
                )
            ],
            f"expert-adapted representation (Delta C@3, {regime})",
        )
        interaction = []
        for s in seeds:
            values = (
                cov(regime, "trainable_on", s),
                cov(regime, "trainable_off", s),
                cov(regime, "frozen_on", s),
                cov(regime, "frozen_off", s),
            )
            if None not in values:
                interaction.append((values[0] - values[1]) - (values[2] - values[3]))
        effects[f"interaction_{regime}"] = s11.paired_stats(
            interaction, f"interaction term (Delta C@3, {regime})"
        )

    # Coverage plus the representation-cost readings, always together.
    per_arm = {}
    for regime in REGIMES:
        for arm in ARMS:
            rows = [c for c in cells if c["regime"] == regime and c["arm"] == arm]
            if not rows:
                continue
            per_arm[f"{regime}/{arm}"] = {
                "accuracy": s11.paired_stats([r["accuracy"] for r in rows], "accuracy"),
                "coverage_at_3": s11.paired_stats(
                    [r["routing"]["coverage"].get("3") for r in rows], "C@3"
                ),
                "conditional_oracle_at_3": s11.paired_stats(
                    [r["routing"]["conditional_oracle"].get("3") for r in rows],
                    "conditional oracle@3",
                ),
                "ceiling_at_3": s11.paired_stats(
                    [r["ceiling"].get("3") for r in rows], "ceiling@3"
                ),
                "n": len(rows),
            }
    effects["per_arm"] = per_arm
    effects["westfall_young"] = s11.westfall_young(
        {
            name: stats["per_seed"]
            for name, stats in effects.items()
            if isinstance(stats, dict) and stats.get("n") == len(seeds)
        },
        len(seeds),
    )
    return effects


def guards(cells: list[dict], seeds: list[int]) -> dict:
    """Frozen arms share a bank; the trainable arms record their own."""
    index = _index(cells)
    bank_share = {}
    for regime in REGIMES:
        hashes = {}
        for arm in ARMS:
            values = {
                index[(regime, arm, s)]["bank_hash"]
                for s in seeds
                if (regime, arm, s) in index
            }
            hashes[arm] = values
        shared = [hashes[a] for a in ("frozen_off", "frozen_on", "trainable_off")]
        bank_share[regime] = {
            "frozen_off": len(hashes["frozen_off"]),
            "frozen_on": len(hashes["frozen_on"]),
            "trainable_off": len(hashes["trainable_off"]),
            "trainable_on": len(hashes["trainable_on"]),
            "same_bank_across_frozen_and_trainable_off": bool(
                hashes["frozen_off"]
                and hashes["frozen_off"]
                == hashes["frozen_on"]
                == hashes["trainable_off"]
            ),
            "trainable_on_bank_differs": bool(
                hashes["trainable_on"]
                and hashes["trainable_on"] != hashes["frozen_off"]
            ),
            "shared_hashes_sample": sorted(next(iter(shared)))[:2] if shared else [],
        }
    # The frozen arms must reproduce S11's L3 exactly.
    deltas = {}
    for regime in REGIMES:
        reference = s11_lookup(seeds, regime)
        for seed in seeds:
            cell = index.get((regime, "frozen_off", seed))
            if cell and seed in reference:
                deltas[f"{regime}/{seed}"] = abs(reference[seed] - cell["accuracy"])
    return {
        "bank_identity": bank_share,
        "frozen_off_reproduces_s11": {
            "checked": len(deltas),
            "max_abs_delta": max(deltas.values()) if deltas else None,
        },
    }


_S11_CACHE: dict = {}


def s11_lookup(seeds, regime):
    key = (tuple(seeds), regime)
    if key not in _S11_CACHE:
        cells = json.load(open("results/s11/s11_confirmatory_study.json"))["cells"]
        _S11_CACHE[key] = {
            c["seed"]: c["accuracy"]
            for c in cells
            if c["construct"] == regime
            and c["level"] == "L3_per_task"
            and c["rank"] == OPERATING["rank"]
            and c["protos"] == OPERATING["protos"]
            and c["num_tasks"] == OPERATING["num_tasks"]
        }
    return _S11_CACHE[key]


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
            "model_family": f"L3_per_task+{cell['arm']}",
            "backbone": "vit_b_16+proj768",
            "backbone_pretraining": "imagenet_frozen",
            "readout": "cosine",
            "expert": "residual_adapter",
        },
        metrics={
            "learning": {
                "accuracy": cell["accuracy"],
                "forgetting": 0.0,
                "acc_matrix": cell["acc_matrix"],
            },
            "cost": {"stored_bytes": 0, "total_params": 0},
        },
        provenance={
            "command": "experiments/rr_factorial.py",
            "prereg": "docs/REPRESENTATION_ROUTING_PREREG.md",
            "arm": cell["arm"],
            "regime": cell["regime"],
            "bank_hash": cell["bank_hash"],
            "prototype_hash": cell["prototype_hash"],
            "prototype_source": cell["prototype_source"],
            "coverage": cell["routing"].get("coverage"),
            "conditional_oracle": cell["routing"].get("conditional_oracle"),
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


def _print(hyps: dict) -> None:
    def fmt(stats):
        if not stats or stats.get("n", 0) == 0:
            return "n/a"
        per = " ".join(f"{v:+.4f}" for v in stats["per_seed"])
        return (
            f"{per}   mean {stats['mean']:+.4f} sd {stats['sd']:.4f} "
            f"CI [{stats['ci95'][0]:+.4f},{stats['ci95'][1]:+.4f}]"
        )

    print("\n" + "=" * 104)
    print("REPRESENTATION x ROUTING OBJECTIVE - four-arm factorial")
    print("=" * 104)
    print("\nEFFECTS (paired over seeds, Delta C@3)")
    for name in list(hyps):
        if name in ("per_arm", "westfall_young"):
            continue
        stats = hyps[name]
        if not isinstance(stats, dict) or "per_seed" not in stats:
            continue
        wy = hyps.get("westfall_young", {}).get(name, {})
        print(f"\n  {name}")
        print(f"    {stats['label']}")
        print(f"    per seed: {fmt(stats)}")
        print(
            f"    exact sign p={stats['sign_p']}  permutation p="
            f"{stats['permutation_p']:.4f}  WY p={wy.get('adjusted_p', float('nan')):.4f}"
        )
    print("\nPER ARM (coverage and the representation-cost readings together)")
    for key, arm in hyps.get("per_arm", {}).items():
        print(f"\n  {key}")
        print(f"    accuracy            {fmt(arm['accuracy'])}")
        print(f"    coverage@3          {fmt(arm['coverage_at_3'])}")
        print(f"    conditional oracle@3 {fmt(arm['conditional_oracle_at_3'])}")
        print(f"    ceiling@3           {fmt(arm['ceiling_at_3'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=ARMS)
    parser.add_argument("--regimes", nargs="*", default=REGIMES)
    parser.add_argument("--seeds", default="42,1,2,3,4,5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/rrf")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    study_path = os.path.join(args.out, "rr_factorial_study.json")

    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_func": args.lambda_func,
        "operating": OPERATING,
        "arms": list(args.arms),
        "regimes": list(args.regimes),
        "routing_lambda": ROUTING_LAMBDA,
        "anchor_ema": ANCHOR_EMA,
    }
    cells: list[dict] = []
    done: set[tuple] = set()
    if os.path.exists(study_path) and not args.force:
        previous = json.load(open(study_path))
        if previous.get("recipe") == recipe:
            cells = previous.get("cells", [])
            done = {(c["regime"], c["arm"], c["seed"]) for c in cells}
            print(f"[RRF] resuming: {len(done)} cells recorded", flush=True)

    def save() -> None:
        payload = {
            "schema_version": "1.0",
            "study": "rr_factorial",
            "prereg": "docs/REPRESENTATION_ROUTING_PREREG.md",
            "backbone": "vit_b_16+proj768",
            "recipe": recipe,
            "seeds": seeds,
            "cells": cells,
            "contracts": {
                f"{c['regime']}__{c['arm']}__seed{c['seed']}": _contract(c)
                for c in cells
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
            f"{c['regime']}__{c['arm']}__seed{c['seed']}": _contract(c)
            for c in payload["cells"]
        }
        with open(study_path, "w") as fh:
            json.dump(payload, fh, indent=1)
        _print(payload["hypotheses"])
        print(f"\n[RRF guards] {json.dumps(payload['guards'], indent=1)}")
        return

    grid = [
        {"regime": regime, "arm": arm, "seed": seed}
        for regime in args.regimes
        for arm in args.arms
        for seed in seeds
        if (regime, arm, seed) not in done
    ]
    print(
        f"[RRF] grid: {len(grid)} new cells "
        f"({len(args.regimes)} regimes x {len(args.arms)} arms x {len(seeds)} seeds)",
        flush=True,
    )
    cache: dict = {}
    for cell in grid:
        result = run_cell(
            cell["regime"], cell["seed"], cell["arm"], args, device, cache
        )
        cells.append(result)
        save()
        print(
            f"[RRF] {cell['regime']:10s} {cell['arm']:14s} seed={cell['seed']:<3d} "
            f"acc={result['accuracy'] * 100:6.2f}  C@3="
            f"{(result['routing']['coverage'].get('3') or float('nan')):.4f}  "
            f"cond_oracle@3="
            f"{(result['routing']['conditional_oracle'].get('3') or float('nan')):.4f}",
            flush=True,
        )

    save()
    print(f"[RRF] wrote {study_path}", flush=True)
    payload = json.load(open(study_path))
    _print(payload["hypotheses"])
    print(f"\n[RRF guards] {json.dumps(payload['guards'], indent=1)}")


if __name__ == "__main__":
    main()
