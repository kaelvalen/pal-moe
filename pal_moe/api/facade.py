"""`PalMoE`: the v3 facade over frozen features (vision / cached-feature backends).

Three time scales over one fixed address space (the frozen backbone's output):

    FAST    `write(Example)`  one row in the append-only KV memory; exact delete.
    MEDIUM  `write(Batch)`    one additive float64 contribution to the ridge statistics;
                              order-invariant, exactly subtractable.
    SLOW    `consolidate()`   frozen representation experts trained on the pending
                              medium batches, grouped by a policy (`by_arrival` control,
                              gated `by_confusion`); immutable after the fit.

Prediction composes them: route (parameter-free) -> expert -> readout, then a FAST
memory hit (cosine >= `memory_threshold`) overrides the parametric answer.
"""

from __future__ import annotations

import argparse

import torch

from pal_moe.arch.readouts import mask_unseen
from pal_moe.core.backbones import FrozenFeatureBackbone
from pal_moe.core.hashing import digest, module_digest
from pal_moe.edit.stats import LinearStats, one_hot
from pal_moe.experts.ladder import LadderModel, train_model
from pal_moe.experts.policies import (
    by_arrival,
    by_confusion,
    check_gate,
    confusion_matrix,
)
from pal_moe.memory.kv_store import FastMemory
from pal_moe.router.prototype import PrototypeTaskRouter
from pal_moe.router.ridge_class import RidgeClassRouter

from .guards import GuardConfig, GuardedEditor
from .records import Batch, ConsolidationReport, EditRecord, Example, Prediction

DEFAULT_RECIPE = {
    "epochs": 10,
    "lr": 1e-3,
    "batch_size": 128,
    "lambda_func": 1.0,
    "rank": 8,
    "protos": 1,
    "top_k": 1,
    "seed": 42,
}


class PalMoE(GuardedEditor):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        backbone=None,
        router: str = "ridge_class",
        ridge: float = 1.0,
        memory_threshold: float = 0.9999,
        canary: torch.Tensor | None = None,
        guards: GuardConfig | None = None,
        recipe: dict | None = None,
        device: str | torch.device = "cpu",
    ):
        self.backbone = backbone or FrozenFeatureBackbone(dim)
        super().__init__(self.backbone.base_hash, guards)
        if router not in ("ridge_class", "prototype"):
            raise KeyError(f"unknown router {router!r}")
        self.dim, self.num_classes, self.device = (
            int(dim),
            int(num_classes),
            torch.device(device),
        )
        self.router_name = router
        self.memory_threshold = float(memory_threshold)
        self.memory = FastMemory(self.dim, device=self.device)
        self.stats = LinearStats(
            self.dim, self.num_classes, ridge=ridge, device=self.device
        )
        self.recipe = {**DEFAULT_RECIPE, **(recipe or {})}
        self.canary = (
            None if canary is None else self.backbone.encode(canary.to(self.device))
        )
        self._raw: dict[
            str, tuple[torch.Tensor, torch.Tensor, int]
        ] = {}  # medium id -> (z, y, task)
        self._consolidated_by: dict[str, str] = {}  # medium id -> consolidation id
        self._consolidations: dict[
            str, dict
        ] = {}  # consolidation id -> bank snapshot info
        self.bank: LadderModel | None = None
        self._class_expert: dict[int, int] = {}  # set by consolidation
        self._proto_cache: tuple | None = None

    # -- derived state -------------------------------------------------------

    def _medium_records(self) -> list[EditRecord]:
        return [r for r in self.log if r.kind == "medium"]

    def class_owner(self) -> dict[int, int]:
        """class -> expert/task: the live consolidation's map, else first arrival."""
        if self._class_expert:
            return dict(self._class_expert)
        owner, tasks = {}, {}
        for r in self._medium_records():
            _, y, task = self._raw[r.id]
            t = tasks.setdefault(task, len(tasks))
            for c in torch.unique(y).tolist():
                owner.setdefault(int(c), t)
        return owner

    def router(self):
        if self.router_name == "ridge_class":
            return RidgeClassRouter(self.stats, self.class_owner())
        if self.bank is not None:
            return PrototypeTaskRouter(self.bank.router)
        return self._prototype_from_log()

    def _prototype_from_log(self) -> PrototypeTaskRouter:
        """Class means over the live medium edits (rebuilt, never downdated)."""
        key = tuple(r.id for r in self._medium_records())
        if self._proto_cache is not None and self._proto_cache[0] == key:
            return self._proto_cache[1]
        owner = self.class_owner()
        r = PrototypeTaskRouter.empty(
            self.dim, self.num_classes, max(1, len(set(owner.values()))), self.device
        )
        sums = torch.zeros(
            self.num_classes, self.dim, dtype=torch.float64, device=self.device
        )
        counts = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)
        for rec in self._medium_records():
            z, y, _ = self._raw[rec.id]
            sums.index_add_(0, y, z.double())
            counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float64))
        for c, t in owner.items():
            r.router.register_class(c, t, (sums[c] / counts[c]).float())
        self._proto_cache = (key, r)
        return r

    def _routers(self) -> list:
        return [self.router()]

    # -- write paths ---------------------------------------------------------

    def write(self, item, path: str | None = None) -> EditRecord:
        """Learn now. `Example` -> FAST memory row; `Batch` -> MEDIUM statistics."""
        return self._guarded_write(item, path)

    def _apply(self, item, path):
        path = path or ("medium" if isinstance(item, Batch) else "fast")
        if path == "fast":
            ex = item if isinstance(item, Example) else Example(*item)
            key = self.backbone.encode(
                torch.as_tensor(ex.x, device=self.device).float()
            )[0]
            value = int(ex.y)
            h = digest(key, value)
            rid = self._next_id("fast", h)
            self.memory.write(rid, key, value)
            return EditRecord(rid, "fast", h, meta={"value": value}), None
        if path == "medium":
            b = item if isinstance(item, Batch) else Batch(*item)
            z = self.backbone.encode(torch.as_tensor(b.x, device=self.device).float())
            y = torch.as_tensor(b.y, device=self.device).long().reshape(-1)
            task = b.task if b.task is not None else f"arrival-{self._seq + 1}"
            h = digest(z, y)
            rid = self._next_id("medium", h)
            c = self.stats.contribution(rid, z, one_hot(y, self.num_classes))
            self.stats.add(c)
            self._raw[rid] = (z, y, task)
            return EditRecord(
                rid,
                "medium",
                h,
                meta={
                    "n": int(y.numel()),
                    "task": task,
                    "classes": sorted(set(y.tolist())),
                },
            ), c
        raise KeyError(f"unknown path {path!r}")

    def _undo(self, record: EditRecord):
        if record.kind == "fast":
            token = self.memory.entry(record.id)
            self.memory.delete(record.id)
            return token
        if record.kind == "medium":
            if record.id in self._consolidated_by:
                raise RuntimeError(
                    f"{record.id} was consolidated by "
                    f"{self._consolidated_by[record.id]}; forget that first"
                )
            c = self.stats.remove(record.id)
            return c, self._raw.pop(record.id)
        if record.kind == "consolidation":
            later = [r for r in self.log if r.kind == "consolidation"]
            if later and later[-1].id != record.id:
                raise RuntimeError(
                    "only the most recent consolidation can be forgotten"
                )
            info = self._consolidations.pop(record.id)
            token = (self.bank, dict(self._class_expert), info)
            self.bank, self._class_expert = info["prev_bank"], info["prev_class_expert"]
            for mid in info["members"]:
                del self._consolidated_by[mid]
            return token
        raise KeyError(record.kind)

    def _redo(self, record: EditRecord, token) -> None:
        if record.kind == "fast":
            key, value, meta = token
            self.memory.write(record.id, key, value, meta)
        elif record.kind == "medium":
            c, raw = token
            self.stats.add(c)
            self._raw[record.id] = raw
        elif record.kind == "consolidation":
            self.bank, self._class_expert, info = token
            self._consolidations[record.id] = info
            for mid in info["members"]:
                self._consolidated_by[mid] = record.id
        self._proto_cache = None

    # -- slow path -----------------------------------------------------------

    def consolidate(
        self,
        policy: str = "by_arrival",
        prereg: str | None = None,
        n_groups: int | None = None,
    ) -> ConsolidationReport:
        """Train frozen experts on the pending medium batches, grouped by `policy`.

        `by_arrival` groups by the batches' `task` (the Stage 1 L3 recipe, run through
        the moved ladder code, so a fresh consolidation of the E-TID2 tasks reproduces
        its bank bitwise). `by_confusion` is gated behind `docs/P2_BOUND_PREREG.md`.
        """
        check_gate(policy, prereg)
        pending = [
            r for r in self._medium_records() if r.id not in self._consolidated_by
        ]
        if not pending:
            raise RuntimeError("nothing to consolidate")
        z = torch.cat([self._raw[r.id][0] for r in pending])
        y = torch.cat([self._raw[r.id][1] for r in pending])
        if policy == "by_arrival":
            order, per_task = [], {}
            for r in pending:
                t = self._raw[r.id][2]
                if t not in per_task:
                    order.append(t)
                per_task.setdefault(t, set()).update(self._raw[r.id][1].tolist())
            groups = by_arrival([sorted(per_task[t]) for t in order])
        elif policy == "by_confusion":
            logits = self.stats.predict(z)
            conf = confusion_matrix(logits.argmax(-1).cpu(), y.cpu(), self.num_classes)
            classes = sorted(set(y.tolist()))
            k = n_groups or len({self._raw[r.id][2] for r in pending})
            groups = by_confusion(conf, k, classes=classes, seed=self.recipe["seed"])
        else:
            raise KeyError(f"unknown policy {policy!r}")

        tasks = []
        for g in groups:
            mask = torch.isin(y, torch.tensor(g, device=y.device))
            split = (z[mask].cpu(), y[mask].cpu())
            tasks.append(
                {
                    "task_id": len(tasks),
                    "classes": list(g),
                    "splits": {"train": split, "val": split, "test": split},
                }
            )

        rec = self.recipe
        h = digest(policy, [r.id for r in pending], groups, sorted(rec.items()))
        rid = self._next_id("consolidation", h)
        prev_bank, prev_map = self.bank, dict(self._class_expert)
        args = argparse.Namespace(
            epochs=rec["epochs"],
            lr=rec["lr"],
            batch_size=rec["batch_size"],
            lambda_func=rec["lambda_func"],
            seed=rec["seed"],
        )
        cell = {
            "seed": rec["seed"],
            "rank": rec["rank"],
            "protos": rec["protos"],
            "top_k": rec["top_k"],
        }
        if prev_bank is not None:
            raise NotImplementedError(
                "incremental consolidation onto an existing bank is not implemented yet; "
                "forget the previous consolidation and consolidate everything pending"
            )
        with torch.enable_grad():
            bank = train_model("L3_per_task", tasks, cell, args, self.device)
        frozen = 0
        for e in bank.experts:
            for p in e.parameters():
                p.requires_grad_(False)
                frozen += p.numel()
        for p in bank.readout.parameters():
            p.requires_grad_(False)
            frozen += p.numel()

        before = self._outputs_digest()
        pre_w = self._weights_digest()
        self.bank = bank
        self._class_expert = {c: i for i, g in enumerate(groups) for c in g}
        self._consolidations[rid] = {
            "prev_bank": prev_bank,
            "prev_class_expert": prev_map,
            "members": [r.id for r in pending],
            "groups": groups,
        }
        for r in pending:
            self._consolidated_by[r.id] = rid
        record = EditRecord(
            rid, "consolidation", h, meta={"policy": policy, "groups": groups}
        )
        self.log.append(record)
        record.purity_report = self.check_purity()
        after = self._outputs_digest()
        record.locality_report = self.locality(
            before[1:], after[1:], self._epsilon("consolidation")
        )
        record.order_hash, record.order_report = self._order_report()
        if self.guards.trial_reversibility:
            record.reversibility_report = self._trial_reversibility(
                record, pre_w, before[0], after[0]
            )
            if not record.reversibility_report["pass"]:
                self._violate(
                    "reversibility", record.reversibility_report, record, rollback=True
                )
        if not record.locality_report["pass"]:
            self._violate("locality", record.locality_report, record, rollback=True)
        self._remember_state()
        return ConsolidationReport(record, policy, groups, len(bank.experts), frozen)

    # -- read path -----------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        x,
        k: int = 1,
        oracle_expert: torch.Tensor | None = None,
        use_memory: bool = True,
        encoded: bool = False,
    ) -> Prediction:
        z = (
            x
            if encoded
            else self.backbone.encode(torch.as_tensor(x, device=self.device).float())
        )
        router = self.router()
        medium = self.stats.predict(z) if self.stats.edit_ids else None
        ids = scores = None
        if self.router_name == "ridge_class" and medium is not None:
            ids, scores = router.top_k(z, k=max(k, 1), class_logits=medium)
        elif self.router_name == "prototype" and (
            self.bank is not None or self.stats.edit_ids
        ):
            ids, scores = router.top_k(z, k=max(k, 1))
        if self.bank is not None:
            chosen = oracle_expert if oracle_expert is not None else ids[:, 0]
            logits = mask_unseen(
                self.bank.readout.predict(self.bank.apply_experts(z, chosen)),
                self.bank.seen,
            )
            source = ["experts"] * z.size(0)
        elif medium is not None:
            logits, source = medium, ["medium"] * z.size(0)
        else:
            logits = torch.zeros(z.size(0), self.num_classes, device=self.device)
            source = ["none"] * z.size(0)
        labels = logits.argmax(-1)
        hits = [[] for _ in range(z.size(0))]
        if use_memory and len(self.memory):
            hits = self.memory.lookup(z, k=1, threshold=self.memory_threshold)
            labels = labels.clone()
            for i, h in enumerate(hits):
                if h:
                    labels[i] = int(h[0].value)
                    source[i] = "memory"
        return Prediction(
            labels,
            logits,
            ids,
            scores,
            source,
            hits,
            medium.argmax(-1) if medium is not None else None,
        )

    # -- guard hooks ----------------------------------------------------------

    def _canary_outputs(self):
        if self.canary is None:
            empty = torch.zeros(0)
            return empty, empty.long()
        p = self.predict(self.canary, encoded=True)
        return p.logits.double().cpu(), p.labels.cpu()

    def _weights_digest(self) -> str:
        bank = (
            "none"
            if self.bank is None
            else digest(
                [module_digest(e) for e in self.bank.experts],
                module_digest(self.bank.readout),
                module_digest(self.bank.router),
            )
        )
        return digest(
            self.memory.state_digest(),
            self.stats.state_digest(),
            bank,
            sorted(self._class_expert.items()),
        )

    def _order_report(self):
        order = self.stats.canonical_order()
        rep = self.stats.order_report(self.canary) if order else {}
        return digest(order), rep
