"""
E2: evidence-producing experts. docs/E2_EVIDENCE_PREREG.md

    h_t = normalize(W_t E_t(z))         W_t : dim -> 128, bias=False
    s_t = (P z) . h_t                   P   : dim -> 128, shared
    e*  = argmax_t s_t                  winner-take-all, unchanged
    l   = g(h_e*)                       g   : 128 -> classes, one shared readout

Training contract, per task: append raw-z class prototypes, freeze the set for the
task, then every optimizer step computes L_task (current examples through the
owner expert, W_t and g) plus L_evidence (ALL stored prototypes through ALL seen
experts and W, shared P), averaged - never summed - over prototypes. All seen W
are trained; previous adapters are frozen and absent from the optimizer.

L_evidence is a function of W and P only: g never enters it.

Usage:  python experiments/e2_evidence.py --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import s10_scaling  # noqa: E402
import s11_confirmatory as s11  # noqa: E402

SOURCE_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"
REGIMES = ["coherent", "dispersed"]
ARMS = ["e0", "e2"]
OPERATING = {"rank": 8, "protos": 1, "num_tasks": 20, "top_k": 1}
D_E = 128
GAMMA = 0.2


def normalize(x):
    return F.normalize(x, dim=-1)


class E2Model:
    """Evidence experts, a shared query and one shared classifier on evidence."""

    def __init__(self, dim, num_classes, args, device):
        self.dim = dim
        self.num_classes = num_classes
        self.device = device
        self.args = args
        self.experts = []
        self.W = []
        self.P = torch.nn.Linear(dim, D_E, bias=False).to(device)
        self.readout = s11.s2_ladder.build_readout(
            "cosine", dim=D_E, num_classes=num_classes
        ).to(device)
        self.seen = []
        self.prototypes = []  # (z_p raw, owner expert)
        self.guard = {}

    def add_task(self, task, task_index):
        builder = s11.s2_ladder.build_expert
        self.experts.append(
            builder("residual_adapter", dim=self.dim, rank=self.args.rank).to(
                self.device
            )
        )
        self.W.append(torch.nn.Linear(self.dim, D_E, bias=False).to(self.device))
        for expert in self.experts[:-1]:
            for param in expert.parameters():
                param.requires_grad_(False)
        feats, labels = task["splits"]["train"]
        feats, labels = feats.to(self.device), labels.to(self.device)
        with torch.no_grad():
            for c in task["classes"]:
                mask = labels == c
                if bool(mask.any()):
                    self.prototypes.append((feats[mask].mean(dim=0), task_index))

    def train(self, tasks, seed, hooks=None):
        """The single training path. `hooks` observes; it never trains.

        `on_task_start(model, t) -> state` runs after the task's prototypes are
        appended and before training, `on_task_end(model, t, state)` after it.
        With `hooks=None` this is byte-identical to the un-instrumented path.
        """
        hooks = hooks or {}
        for t, task in enumerate(tasks):
            self.add_task(task, t)
            state = hooks.get("on_task_start")(self, t) if hooks else None
            self.train_task(task, t, seed)
            if hooks.get("on_task_end"):
                hooks["on_task_end"](self, t, state)

    def train_task(self, task, t, seed):
        """One task of the pinned contract; identical for E2 and the ablation."""
        self.seen = sorted(set(self.seen) | set(task["classes"]))
        # AC1 (`p_alignment="consolidated"`): the shared query trains on task 0
        # and is frozen from the first task boundary on - the exact analogue of
        # C0's treatment of W. "plastic" is the pinned contract, unchanged.
        p_mode = getattr(self.args, "p_alignment", "plastic")
        if p_mode == "consolidated" and t > 0:
            for param in self.P.parameters():
                param.requires_grad_(False)
        trainable = [p for p in self.P.parameters() if p.requires_grad]
        trainable += list(self.readout.parameters())
        for w in self.W:
            trainable += [p for p in w.parameters() if p.requires_grad]
        for param in self.experts[-1].parameters():
            param.requires_grad_(True)
        # C0: only the current evidence projection is trained; the older ones
        # are frozen exactly like the older adapters. C1 (the pinned E2
        # contract) leaves them trainable.
        freeze_old = getattr(self.args, "w_alignment", "all") == "current"
        for w in self.W[:-1]:
            for param in w.parameters():
                param.requires_grad_(not freeze_old)
        trainable = [p for p in self.P.parameters() if p.requires_grad]
        trainable += list(self.readout.parameters())
        for w in self.W:
            trainable += [p for p in w.parameters() if p.requires_grad]
        trainable += list(self.experts[-1].parameters())
        optimizer = torch.optim.Adam(trainable, lr=self.args.lr)
        handle = s11.s2_ladder.trainable_hook(self.readout, task["classes"])
        feats, labels = task["splits"]["train"]
        feats, labels = feats.to(self.device), labels.to(self.device)
        generator = torch.Generator().manual_seed(seed + t)
        checked = False
        for _ in range(self.args.epochs):
            for z, y in s11.s2_ladder.iter_batches(
                feats, labels, self.args.batch_size, generator
            ):
                z, y = z.to(self.device), y.to(self.device)
                h = normalize(self.W[t](self.experts[t].transform(z)))
                logits = s11.s2_ladder.mask_unseen(self.readout.predict(h), self.seen)
                l_task = F.cross_entropy(logits, y)
                l_evidence = self._evidence_loss(t)
                loss = l_task + self.args.evidence_lambda * l_evidence
                optimizer.zero_grad()
                loss.backward()
                if not checked and t > 0:
                    old = [
                        j
                        for j in range(t)
                        if self.W[j].weight.grad is not None
                        and float(self.W[j].weight.grad.abs().sum()) > 0
                    ]
                    current_grad = (
                        self.W[t].weight.grad is not None
                        and float(self.W[t].weight.grad.abs().sum()) > 0
                    )
                    self.guard[f"task{t}"] = {
                        "arm": getattr(self.args, "w_alignment", "all"),
                        "p_alignment": p_mode,
                        "old_W_with_gradient": len(old),
                        "old_W_total": t,
                        "current_W_grad_nonzero": current_grad,
                        "old_W_frozen": all(
                            not p.requires_grad
                            for j in range(t)
                            for p in self.W[j].parameters()
                        ),
                        "P_frozen": all(
                            not p.requires_grad for p in self.P.parameters()
                        ),
                        "P_in_optimizer": sum(
                            1
                            for p in trainable
                            if any(p is q for q in self.P.parameters())
                        ),
                        "P_grad_nonzero": any(
                            p.grad is not None and float(p.grad.abs().sum()) > 0
                            for p in self.P.parameters()
                        ),
                        "previous_experts_in_optimizer": sum(
                            1
                            for p in trainable
                            if any(
                                p is q
                                for e in self.experts[:-1]
                                for q in e.parameters()
                            )
                        ),
                        "previous_experts_frozen": all(
                            not p.requires_grad
                            for e in self.experts[:-1]
                            for p in e.parameters()
                        ),
                    }
                    checked = True
                optimizer.step()
        handle.remove()

    def _evidence_loss(self, task_index=None, rows=False):
        """Mean over stored prototypes of a CE over all seen experts. No g.

        Batched: the pinned contract is full-batch over every prototype at every
        optimizer step, so the prototypes are stacked once per task and scored by
        each expert in one batched call. The arithmetic is identical to a
        per-prototype loop; only the Python overhead changes.

        `w_alignment="owner_only"` (the intervention arm) detaches the projected
        feature of an old expert on prototype rows it does not own, so an old W_j
        receives only its own task's evidence gradient; the values are untouched.
        `rows=True` exposes the per-prototype cross-entropies for the audit only.
        """
        if not self.prototypes:
            return torch.zeros((), device=self.device)
        z = torch.stack([p for p, _ in self.prototypes])  # [P, dim]
        owners = torch.tensor([o for _, o in self.prototypes], device=self.device)
        cut = getattr(self.args, "w_alignment", "all") == "owner_only"
        scores = []
        for j in range(len(self.experts)):
            h = normalize(self.W[j](self.experts[j].transform(z)))  # [P, D_E]
            q = normalize(self.P(z))  # [P, D_E]
            if cut and task_index is not None and j < task_index:
                keep = (owners == j).unsqueeze(-1)
                h = torch.where(keep, h, h.detach())
            scores.append((q * h).sum(dim=-1))  # [P]
        stacked = torch.stack(scores, dim=1)  # [P, T_t]
        if rows:
            return F.cross_entropy(stacked, owners, reduction="none"), owners
        return F.cross_entropy(stacked, owners)

    @torch.no_grad()
    def evaluate(self, tasks):
        correct = total = cov_hits = cov_n_total = oracle_correct = 0
        per_task_coverage = []
        per_task_accuracy = []
        for task in tasks:
            feats, labels = task["splits"]["test"]
            feats, labels = feats.to(self.device), labels.to(self.device)
            q = self.P(feats.float())
            scores = []
            for j in range(len(self.experts)):
                adapted = self.experts[j].transform(feats)
                h = normalize(self.W[j](adapted))
                # per-sample dot product, not the full [B, B] matrix
                scores.append((normalize(q) * h).sum(dim=-1))
            s = torch.stack(scores, dim=1)
            pick = s.argmax(dim=1)
            adapted = self._gather(feats, pick)
            logits = s11.s2_ladder.mask_unseen(self.readout.predict(adapted), self.seen)
            hit = logits.argmax(dim=1) == labels
            correct += int(hit.sum())
            total += int(labels.numel())
            top3 = s.topk(min(3, s.size(1)), dim=1).indices
            covered = (top3 == int(task["task_id"])).any(dim=1)
            cov_n_total += int(covered.sum())
            cov_hits += int(hit[covered].sum())
            per_task_coverage.append(float(covered.float().mean()))
            per_task_accuracy.append(float(hit.float().mean()))
            # oracle path: the owner expert's evidence, same readout
            owner = int(task["task_id"])
            h_owner = normalize(self.W[owner](self.experts[owner].transform(feats)))
            logits_o = s11.s2_ladder.mask_unseen(
                self.readout.predict(h_owner), self.seen
            )
            oracle_correct += int((logits_o.argmax(dim=1) == labels).sum())
        return {
            "accuracy": correct / max(total, 1),
            "coverage_at_3": float(np.mean(per_task_coverage)),
            "oracle_accuracy": oracle_correct / max(total, 1),
            "conditional_oracle_at_3": (
                (cov_hits / max(cov_n_total, 1)) if cov_n_total else None
            ),
            "ceiling_at_3": (
                float(np.mean(per_task_coverage)) * cov_hits / max(cov_n_total, 1)
                if cov_n_total
                else None
            ),
            "per_task_accuracy": per_task_accuracy,
        }

    def _gather(self, feats, pick):
        out = torch.zeros((feats.size(0), D_E), device=self.device)
        for j in torch.unique(pick).tolist():
            mask = pick == j
            h = normalize(self.W[int(j)](self.experts[int(j)].transform(feats[mask])))
            out[mask] = h
        return out


def run_e2(regime, seed, args, device):
    _, source = s11.s2_ladder.load_tasks(SOURCE_CACHE)
    tasks = s11.s6b_difficulty.build_construction(source, regime)
    dim = int(tasks[0]["splits"]["train"][0].size(1))
    num_classes = sum(len(t["classes"]) for t in tasks)
    args.seed = seed
    s11.s2_ladder.set_seed(seed)
    model = E2Model(dim, num_classes, args, device)
    model.train(tasks, seed)
    metrics = model.evaluate(tasks)
    return {
        **metrics,
        "guard": model.guard,
        "d_e": D_E,
        "W_bias": any(w.bias is not None for w in model.W),
    }


def run_e0(regime, seed, args, device):
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
    accuracy = s11.evaluate(model, "L3_per_task", tasks)["accuracy"]
    routing = s10_scaling.routing_report(model, tasks)
    return {"accuracy": accuracy, "coverage_at_3": routing.get("coverage", {}).get("3")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    # Development diagnostic only; the confirmatory contract fixes lambda = 1.0
    parser.add_argument("--evidence_lambda", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=OPERATING["rank"])
    parser.add_argument("--max_experts", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--w_alignment",
        choices=["current", "all", "owner_only"],
        default="all",
        help=(
            "coupling arm: current = freeze old W (C0); all = pinned E2 (C1); "
            "owner_only = old W see only their own task's evidence gradient"
        ),
    )
    parser.add_argument(
        "--p_alignment",
        choices=["plastic", "consolidated"],
        default="plastic",
        help=(
            "AC1 arm: plastic = the shared query P trains on every task (the "
            "pinned contract); consolidated = P trains on task 0 only, then "
            "frozen and absent from the optimizer for every t >= 1"
        ),
    )
    parser.add_argument("--out", default="results/e2")
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    study_path = os.path.join(args.out, "e2_evidence_study.json")
    recipe = {
        "epochs": args.epochs,
        "lr": args.lr,
        "d_e": D_E,
        "gamma": GAMMA,
        "operating": OPERATING,
        "regimes": REGIMES,
        "arms": ARMS,
    }
    cells = []
    done = set()
    if os.path.exists(study_path):
        prev = json.load(open(study_path))
        if prev.get("recipe") == recipe:
            cells = prev.get("cells", [])
            done = {(c["regime"], c["arm"], c["seed"]) for c in cells}

    def save():
        json.dump(
            {
                "schema_version": "1.0",
                "study": "e2_evidence",
                "prereg": "docs/E2_EVIDENCE_PREREG.md",
                "recipe": recipe,
                "seeds": seeds,
                "cells": cells,
                "guards": guards(cells, seeds),
            },
            open(study_path, "w"),
            indent=1,
        )

    for regime in REGIMES:
        for seed in seeds:
            for arm in ARMS:
                if (regime, arm, seed) in done:
                    continue
                if arm == "e0":
                    result = run_e0(regime, seed, args, device)
                else:
                    result = run_e2(regime, seed, args, device)
                cells.append({"regime": regime, "arm": arm, "seed": seed, **result})
                save()
                print(
                    f"[E2] {regime:10s} {arm} seed={seed:<3d} "
                    f"acc={result['accuracy']*100:6.2f} "
                    f"C@3={(result['coverage_at_3'] or float('nan')):.4f}",
                    flush=True,
                )
    save()
    print("[E2] guards:", json.dumps(guards(cells, seeds), indent=1)[:700])


def guards(cells, seeds):
    index = {(c["regime"], c["arm"], c["seed"]): c for c in cells}
    anchor = {}
    for regime in REGIMES:
        for seed in seeds:
            e0 = index.get((regime, "e0", seed))
            ref = s11_reference(regime, seed)
            if e0 and ref is not None:
                anchor[f"{regime}/{seed}"] = abs(ref - e0["accuracy"])
    grad = {}
    for c in cells:
        if c["arm"] == "e2" and c.get("guard"):
            for task, row in c["guard"].items():
                grad[f"{c['regime']}/{c['seed']}/{task}"] = row
    ok = (
        all(
            r["old_W_with_gradient"] == r["old_W_total"]
            and r["previous_experts_in_optimizer"] == 0
            and r["previous_experts_frozen"]
            for r in grad.values()
        )
        if grad
        else None
    )
    return {
        "e0_anchor": {
            "checked": len(anchor),
            "max_abs_delta": max(anchor.values()) if anchor else None,
            "veto_passed": bool(anchor and max(anchor.values()) <= 1e-6),
        },
        "all_W_gradient_and_freeze": {
            "tasks_checked": len(grad),
            "veto_passed": ok,
            "detail": dict(list(grad.items())[:3]),
        },
        "d_e_and_bias": {
            "d_e": D_E,
            "any_W_has_bias": any(
                bool(c.get("W_bias")) for c in cells if c["arm"] == "e2"
            ),
        },
    }


_S11 = {}


def s11_reference(regime, seed):
    if regime not in _S11:
        rows = json.load(open("results/s11/s11_confirmatory_study.json"))["cells"]
        _S11[regime] = {
            c["seed"]: c["accuracy"]
            for c in rows
            if c["construct"] == regime
            and c["level"] == "L3_per_task"
            and c["rank"] == OPERATING["rank"]
            and c["protos"] == OPERATING["protos"]
            and c["num_tasks"] == OPERATING["num_tasks"]
        }
    return _S11[regime].get(seed)


if __name__ == "__main__":
    main()
