"""
S2 - the complexity ladder under one fixed recipe (docs/STAGE1_PLAN.md).

One backbone, one dataset, one protocol. The only thing that changes between
rows is the model's complexity, and the only thing that changes between columns
is the seed. Nothing is tuned per level: if a rung loses it is reported, and a
sensitivity study would be a separate experiment.

    L0    identity expert   + NCM                     minimum baseline
    L1    identity expert   + ridge / linear          cost of a learned readout
    L2a   1 shared adapter  + readout, JOINT          capacity ceiling (NOT a CL result)
    L2b   1 shared adapter  + readout, sequential     sharing interference
    L3    per-task adapters + readout, prototype router   modular isolation
    L4    per-task adapters + readout, oracle router      routing ceiling
    ceiling_joint_probe      frozen features, joint readout (not CL)

Fixed across every level: the feature cache (byte-identical tensors), the task
partition, the class order, the training budget, the evaluation protocol and
the memory quota. `--seed` is the only axis.

Everything is composed through the S1 registries (`pal_moe.arch`), so the ladder
is a config rather than a branch in a training loop, and every row emits an S0
measurement-contract record.

Usage:
    python experiments/s2_ladder.py --seeds 42 1 2 --device cuda
    python experiments/s2_ladder.py --levels L0_ncm L3_per_task --seeds 42
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from pal_moe.arch import (
    PrototypeRouter,
    build_expert,
    build_readout,
    describe,
    mask_unseen,
    trainable_hook,
)
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks

DEFAULT_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"

# Readouts with no trainable parameters are fitted in closed form / by
# registration; the rest are trained by the same outer loop as the expert.
CLOSED_FORM_READOUTS = ("ncm", "ridge")


# ---------------------------------------------------------------------------
# level specification
# ---------------------------------------------------------------------------


@dataclass
class LevelSpec:
    name: str
    expert: str  # registry name, "" for no expert
    experts_per_task: int  # 0 none, -1 one shared, 1 one per task
    readout: str
    router: str  # "none" | "prototype" | "oracle"
    joint: bool = False  # train on all tasks at once (control condition)
    note: str = ""


LADDER: list[LevelSpec] = [
    LevelSpec("L0_ncm", "", 0, "ncm", "none", note="minimum baseline, training-free"),
    LevelSpec("L1_ridge", "", 0, "ridge", "none", note="closed-form readout"),
    LevelSpec("L1_linear", "", 0, "linear", "none", note="learned linear readout"),
    LevelSpec(
        "L1_cosine",
        "",
        0,
        "cosine",
        "none",
        note="learned cosine readout; the reference readout of L2-L4",
    ),
    LevelSpec(
        "ceiling_joint_probe",
        "",
        0,
        "linear",
        "none",
        joint=True,
        note="not CL: reads every task at once",
    ),
    LevelSpec(
        "L2a_shared_joint",
        "residual_adapter",
        -1,
        "cosine",
        "none",
        joint=True,
        note="capacity ceiling for one adapter, NOT a CL result",
    ),
    LevelSpec(
        "L2b_shared_seq",
        "residual_adapter",
        -1,
        "cosine",
        "none",
        note="one shared adapter under the sequential constraint",
    ),
    LevelSpec(
        "L3_per_task",
        "residual_adapter",
        1,
        "cosine",
        "prototype",
        note="modular isolation with a learned (prototype) router",
    ),
    LevelSpec(
        "L4_oracle",
        "residual_adapter",
        1,
        "cosine",
        "oracle",
        note="same model as L3, task id given: the routing ceiling",
    ),
]

LEVELS_BY_NAME = {spec.name: spec for spec in LADDER}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_tasks(cache_path: str):
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = payload["meta"]
    tasks = []
    for row in payload["tasks"]:
        splits = {
            name: (row["splits"][name][0].float(), row["splits"][name][1].long())
            for name in ("train", "val", "test")
        }
        tasks.append(
            {
                "task_id": int(row["task_id"]),
                "classes": [int(c) for c in row["classes"]],
                "splits": splits,
            }
        )
    # Derived, not required: a cache written by a newer or older producer may
    # not carry the dimension in its meta, and the tensors always know it.
    if not meta.get("feature_dim"):
        meta["feature_dim"] = int(tasks[0]["splits"]["train"][0].size(1))
    return meta, tasks


def iter_batches(feats, labels, batch_size, generator=None):
    perm = torch.randperm(feats.size(0), generator=generator)
    for start in range(0, feats.size(0) - batch_size + 1, batch_size):
        idx = perm[start : start + batch_size]
        yield feats[idx], labels[idx]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# the ladder model
# ---------------------------------------------------------------------------


class LadderModel:
    """`(experts) -> readout` over a frozen cached backbone, plus a router.

    The backbone is the feature cache itself (an identity map over `z`), so
    every level sees byte-identical inputs: the "fixed feature preprocessing"
    requirement of the S2 recipe.
    """

    def __init__(
        self,
        spec: LevelSpec,
        dim: int,
        num_classes: int,
        args,
        device,
        router_slots=None,
    ):
        self.spec = spec
        self.dim = dim
        self.num_classes = num_classes
        # The router may need more slots than there are classes: under Domain-IL
        # every task carries the same labels, so keying prototypes by class id
        # makes each task overwrite the previous one's prototypes and the router
        # collapses to a single expert (measured: reuse 0.25, MI 0.0000). Keying
        # by (task, class) instead keeps one prototype per domain-class pair.
        self.router_slots = int(router_slots or num_classes)
        self.device = device
        self.args = args
        self.seen: list[int] = []
        self.experts: list[torch.nn.Module] = []
        self.readout = build_readout(spec.readout, dim=dim, num_classes=num_classes).to(
            device
        )
        self.router: PrototypeRouter | None = None
        self.class_expert: dict[int, int] = {}
        self.proto_z: list[torch.Tensor] = []
        self.proto_p: list[torch.Tensor] = []
        self.proto_task: list[int] = []
        self.active_classes: list[int] = []
        self.optimizer_steps = 0

    # -- experts ---------------------------------------------------------

    def _new_expert(self):
        return build_expert(self.spec.expert, dim=self.dim, rank=self.args.rank).to(
            self.device
        )

    def apply_experts(self, z: torch.Tensor, task_ids) -> torch.Tensor:
        if not self.experts:
            return z
        if self.spec.experts_per_task == -1:
            return self.experts[0].transform(z)
        if task_ids is None:
            return z
        out = z.clone()
        for t in torch.unique(task_ids).tolist():
            if not (0 <= int(t) < len(self.experts)):
                continue
            mask = task_ids == t
            out[mask] = self.experts[int(t)].transform(z[mask])
        return out

    def route_tasks(self, z: torch.Tensor, oracle: bool, task_id):
        """Expert ids per sample. `None` means "no expert applied"."""
        if self.spec.router == "none" or not self.experts:
            return None
        if self.spec.router == "oracle" or oracle:
            if task_id is None:
                raise ValueError("oracle routing needs a task id")
            return torch.full(
                (z.size(0),), int(task_id), dtype=torch.long, device=z.device
            )
        ids, _ = self.router.top_k(z, k=1)
        return ids.squeeze(-1)

    # -- forward ---------------------------------------------------------

    def logits(
        self, z: torch.Tensor, oracle: bool = False, task_id=None
    ) -> torch.Tensor:
        task_ids = self.route_tasks(z, oracle=oracle, task_id=task_id)
        return mask_unseen(
            self.readout.predict(self.apply_experts(z, task_ids)), self.seen
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.logits(z)

    # -- representation (no readout) -------------------------------------

    @torch.no_grad()
    def represent(self, z: torch.Tensor, task_id=None) -> torch.Tensor:
        """The representation the readout would see, i.e. experts applied.

        Used by the forward-transfer probe: it must measure the representation,
        not the current classifier, because the classifier has no rows for a
        task it has not trained on yet.

        The oracle condition has no task id for a task it has never seen, so it
        falls back to the unadapted representation. That is the honest reading:
        oracle routing is an evaluation-time protocol, not information the model
        possesses before the task arrives.
        """
        if self.spec.router == "oracle" and task_id is None:
            return z
        ids = self.route_tasks(z, oracle=False, task_id=task_id)
        return self.apply_experts(z, ids)

    # -- training --------------------------------------------------------

    def fit_task(self, task, task_index: int, joint_data=None) -> None:
        # Which class rows may move in this update. Joint refits open every seen
        # row; the sequential constraint opens only the current task's.
        if joint_data is not None:
            self.active_classes = list(self.seen)
        else:
            self.active_classes = list(task["classes"])
        if self.spec.expert:
            if self.spec.experts_per_task == 1 or not self.experts:
                self.experts.append(self._new_expert())

        if joint_data is not None:
            feats, labels = joint_data
            self._optimize(feats, labels, None, task_index)
            return
        feats, labels = task["splits"]["train"]
        ids = None if self.spec.experts_per_task == -1 else task_index
        self._optimize(feats, labels, ids, task_index)

    def _optimize(self, feats, labels, task_index_or_none, seed_offset: int) -> None:
        trainable = list(self.readout.parameters())
        if self.experts:
            expert = (
                self.experts[-1] if self.spec.experts_per_task == 1 else self.experts[0]
            )
            trainable += list(expert.parameters())
        if not trainable:
            return  # ncm / ridge: fitted in closed form in register_task

        opt = torch.optim.Adam(trainable, lr=self.args.lr)
        handle = trainable_hook(self.readout, self.active_classes)
        gen = torch.Generator().manual_seed(self.args.seed + seed_offset)
        for _ in range(self.args.epochs):
            for z, y in iter_batches(feats, labels, self.args.batch_size, gen):
                z, y = z.to(self.device), y.to(self.device)
                if task_index_or_none is None:
                    ids = None
                elif self.spec.experts_per_task == 1:
                    ids = torch.full(
                        (z.size(0),),
                        len(self.experts) - 1,
                        dtype=torch.long,
                        device=self.device,
                    )
                else:
                    ids = None
                logits = mask_unseen(
                    self.readout.predict(self.apply_experts(z, ids)), self.seen
                )
                loss = F.cross_entropy(logits, y)
                if self.args.lambda_func > 0 and self.experts and self.proto_z:
                    loss = loss + self.args.lambda_func * self._l_func()
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.optimizer_steps += 1
        handle.remove()

    def _l_func(self) -> torch.Tensor:
        """Anchor the registered output on the stored prototypes.

        Evaluated through the same expert path the model would use, so the
        constraint protects the representation it is registered in (the fix
        found in E0: applying no expert here made L_func a no-op for a shared
        adapter and cost 48 points on L2b).
        """
        z = torch.stack(self.proto_z).to(self.device)
        p = torch.stack(self.proto_p).to(self.device)
        if self.spec.experts_per_task == 1:
            ids = torch.tensor(self.proto_task, device=self.device)
        else:
            ids = None
        return F.cross_entropy(
            mask_unseen(self.readout.predict(self.apply_experts(z, ids)), self.seen), p
        )

    # -- registration ----------------------------------------------------

    @torch.no_grad()
    def register_task(
        self, task, task_index: int, router_offset: int | None = None
    ) -> None:
        """Register a task's prototypes.

        `router_offset` shifts the router's prototype keys by `task_index *
        classes_per_task` so that tasks sharing a label space (Domain-IL) get
        their own prototypes instead of overwriting each other. The readout
        keeps using the true labels either way.
        """
        feats, labels = task["splits"]["train"]
        feats, labels = feats.to(self.device), labels.to(self.device)
        if self.router is None and self.spec.router == "prototype":
            self.router = PrototypeRouter(
                dim=self.dim,
                num_classes=self.router_slots,
                num_experts=max(1, self.args.max_experts),
            ).to(self.device)

        means = {}
        for c in task["classes"]:
            mask = labels == c
            if not bool(mask.any()):
                continue
            means[c] = feats[mask].mean(dim=0)
            self.class_expert[int(c)] = task_index
            if self.router is not None:
                key = (
                    router_offset + task["classes"].index(c)
                    if router_offset is not None
                    else int(c)
                )
                self.router.register_class(key, task_index, means[c])

        for _c, mean in means.items():
            self.proto_z.append(mean.detach().clone())
            self.proto_task.append(task_index)
            ids = (
                torch.tensor([task_index], device=self.device)
                if self.spec.experts_per_task == 1
                else None
            )
            logits = mask_unseen(
                self.readout.predict(self.apply_experts(mean.unsqueeze(0), ids)),
                self.seen,
            )
            self.proto_p.append(F.softmax(logits, dim=-1).squeeze(0))

        if self.spec.readout in CLOSED_FORM_READOUTS:
            # Fitted after the expert, so a closed-form readout is solved in the
            # same space the trained readouts are optimised in.
            self.readout.fit(feats, labels, seen_classes=self.seen)

    # -- evaluation ------------------------------------------------------

    @torch.no_grad()
    def evaluate_task(
        self,
        task,
        oracle: bool = False,
        task_id=None,
        class_masking: bool = False,
    ) -> float:
        """Accuracy on one task's test split.

        `class_masking` restricts the class search space to that task's own
        classes, which is what makes an evaluation **Task-IL** rather than
        Class-IL. It is orthogonal to `oracle` (whether the task id is used for
        *routing*): knowing which task an input belongs to and knowing which
        classes are answerable are two different pieces of information, and S5
        measures them separately (docs/STAGE1_RESULTS.md section 6).
        """
        feats, labels = task["splits"]["test"]
        feats, labels = feats.to(self.device), labels.to(self.device)
        allowed = [int(c) for c in task["classes"]] if class_masking else None
        correct = 0
        for start in range(0, feats.size(0), 512):
            z = feats[start : start + 512]
            y = labels[start : start + 512]
            logits = self.logits(
                z,
                oracle=oracle,
                task_id=task_id if task_id is not None else task["task_id"],
            )
            if allowed is not None:
                logits = mask_unseen(logits, allowed)
            correct += int((logits.argmax(dim=-1) == y).sum())
        return correct / max(feats.size(0), 1)

    @torch.no_grad()
    def routing_stats(self, tasks, k: int = 3) -> dict:
        if self.router is None or not self.experts or self.spec.router == "oracle":
            return {}
        hits = covered_n = covered_correct = total = 0
        for task in tasks:
            feats, labels = task["splits"]["test"]
            feats, labels = feats.to(self.device), labels.to(self.device)
            ids, _ = self.router.top_k(feats, k=k)
            covered = (ids == task["task_id"]).any(dim=-1)
            correct = self.logits(feats).argmax(dim=-1) == labels
            hits += int(covered.sum())
            covered_n += int(covered.sum())
            covered_correct += int(correct[covered].sum())
            total += feats.size(0)
        return {
            f"task_recall_at_{k}": hits / max(total, 1),
            "acc_covered": covered_correct / max(covered_n, 1),
        }

    def cost(self) -> dict:
        params = int(sum(p.numel() for p in self.readout.parameters()))
        if self.experts:
            params += int(sum(p.numel() for p in self.experts[0].parameters()))
        flops = 4 * self.dim * self.num_classes  # readout: multiply + add
        if self.experts:
            flops += 4 * self.args.rank * self.dim  # adapter up/down
        if self.router is not None:
            flops += 4 * self.num_classes * self.dim
        return {
            "params": params,
            "flops_forward": flops,
            "optimizer_steps": self.optimizer_steps,
        }


# ---------------------------------------------------------------------------
# one level, one seed
# ---------------------------------------------------------------------------


def run_level(spec: LevelSpec, tasks, meta, args, device) -> dict:
    set_seed(args.seed)
    dim = int(meta["feature_dim"])
    num_classes = sum(len(t["classes"]) for t in tasks)
    model = LadderModel(spec, dim, num_classes, args, device)

    n = len(tasks)
    R = np.zeros((n, n), dtype=np.float32)
    fwt: list[float] = []
    t0 = time.time()

    if spec.joint:
        all_feats = torch.cat([t["splits"]["train"][0] for t in tasks], dim=0)
        all_labels = torch.cat([t["splits"]["train"][1] for t in tasks], dim=0)
        model.seen = sorted({c for t in tasks for c in t["classes"]})
        for i, task in enumerate(tasks):
            model.register_task(task, i)
        model.fit_task(None, 0, joint_data=(all_feats, all_labels))
        if spec.readout in CLOSED_FORM_READOUTS:
            model.readout.fit(all_feats, all_labels, seen_classes=model.seen)
        for i in range(n):
            R[n - 1, i] = model.evaluate_task(tasks[i])
    else:
        for t, task in enumerate(tasks):
            # Forward transfer: does the representation built so far help the
            # incoming task? Measured with a closed-form probe (see
            # `forward_transfer`). Task 0 has nothing to transfer from.
            if model.seen:
                _adapted, delta = forward_transfer(model, task, num_classes, dim)
                fwt.append(delta)
            model.seen = sorted(set(model.seen) | set(task["classes"]))
            model.fit_task(task, t)
            model.register_task(task, t)
            for i in range(t + 1):
                R[t, i] = model.evaluate_task(tasks[i])

    T = n - 1
    forget = [
        max(0.0, float(np.max(R[i : T + 1, i])) - float(R[T, i])) for i in range(T)
    ]
    routing = model.routing_stats(tasks)
    cost = model.cost()
    return {
        "level": spec.name,
        "seed": args.seed,
        "note": spec.note,
        "joint": spec.joint,
        "avg_accuracy": float(np.mean(R[T, :])),
        "forgetting": float(np.mean(forget)) if forget else 0.0,
        "fwt": float(np.mean(fwt)) if fwt else None,
        "acc_matrix": R.tolist(),
        "oracle_accuracy": float(np.mean(R[T, :])) if spec.router == "oracle" else None,
        "seconds": round(time.time() - t0, 1),
        "latency_ms": measure_latency(model, tasks[0]["splits"]["test"][0], device),
        **routing,
        **cost,
    }


def forward_transfer(model, task, num_classes: int, dim: int) -> tuple[float, float]:
    """Forward transfer, measured at the REPRESENTATION level.

    The obvious definition - evaluate the model on the incoming task before
    training on it - is degenerate in a growing-head class-incremental setting:
    the task's classes have no head rows yet and are masked, so the accuracy is
    structurally 0 (measured: -3.73% after chance correction, exactly
    `-mean(1/5t)`, identical for every level). What can be measured is whether
    the representation built so far makes the NEXT task easier to learn:

        probe(z) = closed-form ridge on the incoming task, fitted on
                   `model.represent(train)` and tested on
                   `model.represent(test)`

    `fwt = probe_adapted - probe_raw`, where `probe_raw` uses the frozen
    backbone with no expert. Positive means the adaptation so far helps a task
    it has never seen; negative means it hurts. Both terms use the same
    closed-form probe, so the comparison carries no optimiser noise.
    """
    feats, labels = task["splits"]["train"]
    test_feats, test_labels = task["splits"]["test"]
    feats, labels = feats.to(model.device), labels.to(model.device)
    test_feats, test_labels = test_feats.to(model.device), test_labels.to(model.device)

    def probe_accuracy(z_train, z_test) -> float:
        probe = build_readout("ridge", dim=dim, num_classes=num_classes).to(
            model.device
        )
        probe.fit(z_train, labels, seen_classes=sorted(set(labels.tolist())))
        return float(
            (probe.predict(z_test).argmax(dim=-1) == test_labels).float().mean()
        )

    adapted = probe_accuracy(model.represent(feats), model.represent(test_feats))
    raw = probe_accuracy(feats, test_feats)
    return adapted, adapted - raw


def measure_latency(
    model, feats: torch.Tensor, device, n: int = 128, repeats: int = 20
) -> float:
    """Median per-sample forward latency in milliseconds (best effort)."""
    if device.type != "cuda":
        return float("nan")
    z = feats[:n].to(device)
    with torch.no_grad():
        for _ in range(3):
            model.logits(z, task_id=0)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.logits(z, task_id=0)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0 / z.size(0))
    return float(statistics.median(times))


# ---------------------------------------------------------------------------
# study aggregation
# ---------------------------------------------------------------------------


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "n": 1}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "n": len(values),
    }


def aggregate(records: list[dict], baseline: str = "L0_ncm") -> dict:
    by_level: dict[str, list[dict]] = {}
    for record in records:
        by_level.setdefault(record["level"], []).append(record)

    levels = {
        level: {
            "accuracy": _stats([r["avg_accuracy"] for r in rs]),
            "forgetting": _stats([r["forgetting"] for r in rs]),
            "fwt": _stats([r.get("fwt") for r in rs]),
            "params": rs[0]["params"],
            "flops_forward": rs[0]["flops_forward"],
            "latency_ms": _stats([r.get("latency_ms") for r in rs]),
            "task_recall_at_3": _stats([r.get("task_recall_at_3") for r in rs]),
            "acc_covered": _stats([r.get("acc_covered") for r in rs]),
            "oracle_accuracy": _stats([r.get("oracle_accuracy") for r in rs]),
            "optimizer_steps": rs[0].get("optimizer_steps"),
            "seconds": _stats([r.get("seconds") for r in rs]),
            "seeds": [r["seed"] for r in rs],
        }
        for level, rs in by_level.items()
    }
    # Paired per-seed deltas against the baseline (S0 rule R7).
    base_by_seed = {r["seed"]: r["avg_accuracy"] for r in by_level.get(baseline, [])}
    for level, rs in by_level.items():
        deltas = [
            r["avg_accuracy"] - base_by_seed[r["seed"]]
            for r in rs
            if r["seed"] in base_by_seed
        ]
        levels[level]["delta_vs_baseline"] = _stats(deltas)
        levels[level]["wins"] = sum(1 for d in deltas if d > 0)
        levels[level]["paired_n"] = len(deltas)

    # The routing cost is the L4 - L3 pair, computed per seed.
    if "L3_per_task" in by_level and "L4_oracle" in by_level:
        l3 = {r["seed"]: r["avg_accuracy"] for r in by_level["L3_per_task"]}
        l4 = {r["seed"]: r["avg_accuracy"] for r in by_level["L4_oracle"]}
        gaps = [l4[s] - l3[s] for s in l3 if s in l4]
        levels["routing_cost_L4_minus_L3"] = {"accuracy": _stats(gaps), "n": len(gaps)}
    return {"baseline": baseline, "levels": levels}


# ---------------------------------------------------------------------------
# contract emission
# ---------------------------------------------------------------------------


def _memory_bytes(spec: LevelSpec, meta) -> int:
    dim = int(meta["feature_dim"])
    n_classes = 100
    bytes_ = 0
    if spec.readout == "ncm":
        bytes_ += n_classes * dim * 4
    if spec.router == "prototype":
        bytes_ += n_classes * dim * 4
    return int(bytes_)


def _contract(row, spec: LevelSpec, args, meta, tasks) -> dict:
    latent = meta.get("latent_dim")
    arch = meta["encoder_arch"]
    factors = {
        "dataset": meta["dataset"],
        "protocol": "class_il",
        "task_id_at_inference": spec.router == "oracle",
        "num_tasks": len(tasks),
        "classes_per_task": len(tasks[0]["classes"]),
        "class_order": [list(t["classes"]) for t in tasks],
        "seed": args.seed,
        "model_family": row["level"],
        # The projected backbone is named explicitly: sweeping architectures at
        # a fixed latent dimension is the S3 control, and a record that hides
        # the projection would be unreadable later.
        "backbone": f"{arch}+proj{latent}" if latent else arch,
        "backbone_pretraining": (
            "random"
            if meta.get("encoder_weights") == "none"
            else f"{meta['encoder_weights']}_frozen"
        ),
        "readout": spec.readout,
        "readout_estimator": "offline_mean" if spec.readout == "ncm" else None,
        "expert": spec.expert or "none",
    }
    if latent:
        factors["latent_dim"] = int(latent)
        factors["rep_seed"] = meta.get("rep_seed")
    record = build_run_record(
        factors=factors,
        metrics={
            "learning": {
                "accuracy": row["avg_accuracy"],
                "forgetting": row["forgetting"],
                "fwt": row.get("fwt"),
                "acc_matrix": row["acc_matrix"],
            },
            "cost": {
                "stored_bytes": _memory_bytes(spec, meta),
                "total_params": row["params"],
                "flops_forward": row["flops_forward"],
                "latency_ms": row.get("latency_ms"),
                "optimizer_steps": row["optimizer_steps"],
            },
            "modular": {
                "expert_count": 0 if not spec.expert else len(tasks),
                "task_recall_at_k": (
                    {"3": row["task_recall_at_3"]}
                    if "task_recall_at_3" in row
                    else None
                ),
                "oracle_accuracy": row.get("oracle_accuracy"),
            },
        },
        provenance={
            "git_commit": meta.get("git_commit", "unknown"),
            "device": str(args.device),
            "command": "experiments/s2_ladder.py",
        },
    )
    record["blocks"] = satisfied_blocks(record)
    return record


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--seeds", default="42 1 2")
    parser.add_argument("--levels", nargs="*", default=[spec.name for spec in LADDER])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument("--max_experts", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/s2")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    meta, tasks = load_tasks(args.cache)
    meta["git_commit"] = _git_commit()
    print(
        f"[S2] {meta['encoder_arch']}/{meta['encoder_weights']} frozen, "
        f"{len(tasks)} tasks, dim={meta['feature_dim']}, device={device}"
    )
    print(f"[S2] registry: {json.dumps(describe())}")

    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    records, contracts = [], {}
    for name in args.levels:
        spec = LEVELS_BY_NAME[name]
        for seed in seeds:
            args.seed = seed
            row = run_level(spec, tasks, meta, args, device)
            records.append(row)
            contracts[f"{name}__seed{seed}"] = _contract(row, spec, args, meta, tasks)
            print(
                f"[S2] {name:22s} seed={seed:<3d} acc={row['avg_accuracy']*100:6.2f}% "
                f"F={row['forgetting']*100:6.2f}% params={row['params']:8d} "
                f"{row['seconds']:6.1f}s"
            )

    study = aggregate(records)
    os.makedirs(args.out, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "study": "s2_complexity_ladder",
        "factors": {
            "dataset": meta["dataset"],
            "protocol": "class_il",
            "task_id_at_inference": False,
            "backbone": meta["encoder_arch"],
            "backbone_pretraining": f"{meta['encoder_weights']}_frozen",
            "seeds": seeds,
            "fixed": [
                "feature_cache",
                "task_partition",
                "class_order",
                "training_budget",
                "evaluation_protocol",
            ],
        },
        "records": records,
        "contracts": contracts,
        "aggregate": study,
    }
    path = os.path.join(args.out, f"s2_ladder_study{args.tag}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[S2] wrote {path}")
    _print_table(study)


def _print_table(study: dict) -> None:
    print("\n" + "=" * 84)
    print("S2 COMPLEXITY LADDER")
    print("=" * 84)
    print(
        f"{'level':24s} {'acc':>8s} {'std':>7s} {'F':>7s} {'d vs L0':>9s} "
        f"{'params':>9s} {'steps':>7s}"
    )
    for level, s in study["levels"].items():
        if "forgetting" not in s:  # derived rows (e.g. the routing cost)
            continue
        acc = s.get("accuracy") or {}
        d = s.get("delta_vs_baseline") or {}
        print(
            f"{level:24s} {acc.get('mean', float('nan')) * 100:7.2f}% "
            f"{(acc.get('std') or 0) * 100:6.2f}% "
            f"{(s.get('forgetting') or {}).get('mean', float('nan')) * 100:6.2f}% "
            f"{d.get('mean', float('nan')) * 100:+8.2f}% "
            f"{s.get('params', 0):9d} "
            f"{s.get('optimizer_steps', 0):7d}"
        )


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - not a git checkout
        return "unknown"


if __name__ == "__main__":
    main()
