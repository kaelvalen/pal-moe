"""The Stage 1 ladder model: `(experts) -> readout` over a frozen cached backbone.

Moved verbatim from `experiments/s2_ladder.py` (LevelSpec, LADDER, LadderModel,
forward_transfer) and `experiments/s11_confirmatory.py` (train_model, evaluate)
in the v3 restructure, phase 1. Behaviour is frozen: the S11 E0 anchor must stay
bitwise (see `experiments/v3_anchors.py`).
"""

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from pal_moe.arch import (
    PrototypeRouter,
    build_expert,
    build_readout,
    mask_unseen,
    trainable_hook,
)
from pal_moe.core.features import iter_batches, set_seed


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


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in nats over a distribution (same definition as S6)."""
    p = probs[probs > 0]
    return -(p * p.log()).sum()



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
        # S8 resource knobs. `router_prototypes` is the memory axis (how many
        # prototypes per class the router stores), `top_k_experts` the
        # active-compute axis (how many experts are evaluated per sample); the
        # parameter axis is the adapter rank, set through `build_expert`.
        self.router_prototypes = 1
        self.top_k_experts = 1
        self.device = device
        self.args = args
        self.seen: list[int] = []
        self.experts: list[torch.nn.Module] = []
        self.readout = build_readout(spec.readout, dim=dim, num_classes=num_classes).to(
            device
        )
        self.router: PrototypeRouter | None = None
        # Optional extra training term, supplied by a study (S11's factorial
        # uses it for the joint routing objective). `None` is the ladder's
        # unchanged behaviour.
        self.extra_loss = None
        # Optional shared projection applied to every expert's output before the
        # readout (the E1 formulation in EXPERT_FORMULATION_PREREG.md). `None` is
        # the unchanged ladder: E0's path stays byte-for-byte identical.
        self.projection = None
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

    def route_tasks_topk(self, z: torch.Tensor, k: int):
        """Top-k expert ids and mixture weights: the active-compute axis (S8).

        `k = 1` reproduces the hard router exactly, so the first point of that
        sweep is the S2-S6 baseline rather than a new method.
        """
        ids, scores = self.router.top_k(z, k=k)
        temperature = float(getattr(self.router, "temperature", 1.0) or 1.0)
        weights = torch.softmax(scores / temperature, dim=-1)
        return ids, weights

    def apply_experts_mixture(self, z, ids, weights) -> torch.Tensor:
        """`z + sum_j w_j (E_{t_j}(z) - z)`, a convex mixture of adapters."""
        out = z.clone()
        for j in range(ids.size(1)):
            for t in torch.unique(ids[:, j]).tolist():
                if not (0 <= int(t) < len(self.experts)):
                    continue
                mask = ids[:, j] == t
                delta = self.experts[int(t)].transform(z[mask]) - z[mask]
                out[mask] = out[mask] + weights[mask, j].unsqueeze(-1) * delta
        return out

    # -- forward ---------------------------------------------------------

    def logits(
        self, z: torch.Tensor, oracle: bool = False, task_id=None
    ) -> torch.Tensor:
        top_k = int(getattr(self, "top_k_experts", 1) or 1)
        if top_k > 1 and self.router is not None and not oracle:
            ids, weights = self.route_tasks_topk(z, top_k)
            adapted = self.apply_experts_mixture(z, ids, weights)
        else:
            task_ids = self.route_tasks(z, oracle=oracle, task_id=task_id)
            adapted = self.apply_experts(z, task_ids)
        # The E1 formulation: one shared projection for every expert's output,
        # applied immediately before the shared readout.
        if self.projection is not None:
            adapted = self.projection(adapted)
        return mask_unseen(self.readout.predict(adapted), self.seen)

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
        if self.projection is not None:
            trainable += list(self.projection.parameters())
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
                if self.extra_loss is not None:
                    loss = loss + self.extra_loss(z, y, task_index_or_none)
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
        adapted = self.apply_experts(z, ids)
        if self.projection is not None:
            adapted = self.projection(adapted)
        return F.cross_entropy(mask_unseen(self.readout.predict(adapted), self.seen), p)

    # -- registration ----------------------------------------------------

    @staticmethod
    def _class_prototypes(feats: torch.Tensor, k: int) -> list[torch.Tensor]:
        """`k` representatives of one class: the mean when k=1, else k-means.

        Deterministic (5 iterations from a fixed seeded init), because a budget
        sweep must not introduce its own variance source.
        """
        if k <= 1 or feats.size(0) <= k:
            return [feats.mean(dim=0)]
        generator = torch.Generator().manual_seed(0)
        centres = feats[torch.randperm(feats.size(0), generator=generator)[:k]].clone()
        for _ in range(5):
            assign = torch.cdist(feats, centres).argmin(dim=1)
            for j in range(k):
                members = feats[assign == j]
                if members.size(0) > 0:
                    centres[j] = members.mean(dim=0)
        return [centre for centre in centres]

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
                num_classes=self.router_slots
                * max(1, int(getattr(self, "router_prototypes", 1) or 1)),
                num_experts=max(1, self.args.max_experts),
            ).to(self.device)

        per_class_prototypes = max(1, int(getattr(self, "router_prototypes", 1) or 1))
        means = {}
        for c in task["classes"]:
            mask = labels == c
            if not bool(mask.any()):
                continue
            means[c] = feats[mask].mean(dim=0)
            self.class_expert[int(c)] = task_index
            if self.router is not None:
                base_key = (
                    router_offset + task["classes"].index(c)
                    if router_offset is not None
                    else int(c)
                )
                # The memory axis: `k` prototypes per class instead of one mean.
                # k=1 is the mean (the S2-S6 behaviour); larger k buys routing
                # resolution at k times the stored bytes.
                for slot, prototype in enumerate(
                    self._class_prototypes(feats[mask], per_class_prototypes)
                ):
                    self.router.register_class(
                        base_key * per_class_prototypes + slot, task_index, prototype
                    )

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
        shares = []
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
            # Assignment concentration for this task: the fraction of its
            # samples each expert receives. Same definition as S6's unseen-task
            # spread, so the two stages are comparable.
            top1 = ids[:, 0]
            shares.append(
                torch.bincount(top1, minlength=len(self.experts)).float()
                / max(top1.numel(), 1)
            )
        if not shares:
            return {}
        max_share = torch.stack([s.max() for s in shares]).mean()
        entropy = torch.stack([_entropy(s) for s in shares]).mean()
        return {
            f"task_recall_at_{k}": hits / max(total, 1),
            "acc_covered": covered_correct / max(covered_n, 1),
            "expert_max_share": float(max_share),
            "expert_entropy": float(entropy),
            "expert_entropy_normalized": float(
                entropy / float(np.log(max(len(self.experts), 2)))
            ),
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

    def resources(self) -> dict:
        """The three S8 resource axes, measured rather than declared.

        `total_params` is everything that must be stored, `trainable_params` is
        what an update touches (the newest expert plus the readout), and
        `active_params` is what one sample's forward pass evaluates. For a bank
        of per-task experts the first and the last differ by the expert count,
        which is exactly the distinction S8 needs: a 20-expert bank is not 20x
        the per-sample compute.
        """
        readout_params = int(sum(p.numel() for p in self.readout.parameters()))
        readout_buffers = int(sum(b.numel() for b in self.readout.buffers()))
        expert_params = [
            int(sum(p.numel() for p in e.parameters())) for e in self.experts
        ]
        active_experts = min(
            max(int(getattr(self, "top_k_experts", 1) or 1), 1), len(expert_params)
        )
        router_bytes = 0
        if self.router is not None:
            router_bytes = int(
                sum(b.numel() * b.element_size() for b in self.router.buffers())
            )
        expert_bytes = sum(n * 4 for n in expert_params)
        # A closed-form readout's state is its stored sufficient statistics, so it
        # is counted as parameter-equivalents: otherwise ridge would appear to
        # cost nothing while storing 2.9 MB of `A`, `B` and `W`.
        readout_state = readout_params + readout_buffers
        readout_bytes = readout_state * 4
        return {
            "total_params": readout_state + sum(expert_params),
            "trainable_params": readout_params
            + (expert_params[-1] if expert_params else 0),
            "active_params": readout_state
            + active_experts * (expert_params[0] if expert_params else 0),
            "readout_state": readout_state,
            "num_experts": len(expert_params),
            "active_experts": active_experts if expert_params else 0,
            "num_prototypes": (
                int(self.router.counts.gt(0).sum()) if self.router is not None else 0
            ),
            "memory_bytes": readout_bytes + expert_bytes + router_bytes,
            "router_bytes": router_bytes,
            "expert_bytes": expert_bytes,
            "readout_bytes": readout_bytes,
            "flops_forward": self.cost()["flops_forward"],
            "optimizer_steps": self.optimizer_steps,
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


# ---------------------------------------------------------------------------
# S11 train / evaluate (moved from experiments/s11_confirmatory.py)
# ---------------------------------------------------------------------------


def train_model(level: str, tasks: list[dict], cell: dict, args, device):
    dim = int(tasks[0]["splits"]["train"][0].size(1))
    num_classes = sum(len(t["classes"]) for t in tasks)
    spec = LEVELS_BY_NAME[level]
    cell_args = argparse.Namespace(
        rank=cell["rank"],
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=cell["seed"],
        lambda_func=args.lambda_func,
        max_experts=max(20, len(tasks)),
    )
    set_seed(cell["seed"])
    model = LadderModel(
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
    spec = LEVELS_BY_NAME[level]
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
