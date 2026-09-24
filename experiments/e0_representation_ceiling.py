"""
E0 - representation ceiling and adapter headroom on cached frozen features.

Prerequisite for PAL-MoE v2 (docs/PALMOE_V2_SPEC.md section 10, E0). Standalone
on purpose: it does NOT import pal_moe/v2, so it can run before M1 is written.
It answers one question before any v2 package code exists:

    frozen pretrained representation + small residual adapters
    -> is there any headroom above the v1 Pareto row?

Four modes, all on the same class-incremental protocol and the same cached
features (CIFAR-100, 20 tasks x 5 classes, frozen ViT-B/16):

    ncm      nearest-class-mean over per-class means of z_s, zero training.
             The RanPAC-lite / SimpleCIL reference: what the frozen
             representation already gives for free.
    joint    one cosine classifier over all 100 classes trained on all tasks
             jointly. The absolute ceiling of the frozen space (not a CL
             method, it sees every task at once).
    cl       class-incremental cosine classifier: only the current task's rows
             are trained, old rows are frozen, no adapters. The M4 red line
             (spec section 12) measured standalone.
    adapter  `cl` plus one residual adapter per task (always expand; old
             adapters frozen) and L_func on stored prototypes. Rank 0 must
             reproduce `cl`. The gap `adapter(r) - cl` is the marginal value of
             the v2 adapter architecture.

Protocol notes (why this is comparable to the repo tables):

- Routing at eval is a prototype router: the nearest class mean in the FROZEN
  space selects the task whose adapter is applied. This mirrors the v2 design
  (routing reads z_s, prediction reads z_mix) and needs no oracle task id.
  `--oracle_routing` reports the task-incremental upper bound separately.
- Evaluation is class-incremental over the classes seen so far (growing head).
  The v1 tables instead use a fixed 100-way head whose rows are all trained
  with 100-way CE on each task, so v1's early-task accuracies face 95 trained
  distractors while a growing head faces 5(t-1). The two are not directly
  comparable; E0a re-runs v1 at HEAD to get the matching number.

Usage:
    python experiments/e0_representation_ceiling.py --mode all --device cuda
    python experiments/e0_representation_ceiling.py --mode adapter --rank 32
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pal_moe.arch import ResidualAdapter

DEFAULT_CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"


class CosineHead(nn.Module):
    """Growing cosine classifier: logits = scale * <normalize(z), normalize(W)>."""

    def __init__(self, dim: int, num_classes: int, scale: float = 10.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, dim) / dim**0.5)
        self.scale = float(scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.scale * F.normalize(z, dim=-1) @ F.normalize(self.W, dim=-1).t()


class AcceptHead(nn.Module):
    """One acceptance head: r(z) = sigmoid(w . z + b), zero-init and RNG-free.

    Deliberately does not use nn.Linear: its constructor calls
    reset_parameters(), which draws from the global RNG and would shift the
    stream that the adapters are created from, making the rejection run
    incomparable with the baseline.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, dim))
        self.bias = nn.Parameter(torch.full((1,), -2.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.weight, self.bias)


class RejectorBank(nn.Module):
    """Expert acceptance: r_j(z) = sigmoid(w_j . z + b_j), one head per expert.

    Two properties that make this different from a generic known/unknown
    detector, and both matter for the failure E0 measured (F5: a candidate
    expert scores other tasks' data highly):

    - expert-conditioned: the decision is "should THIS expert answer", not
      "is this input familiar";
    - future-negative-aware: the bank is re-fit on the FULL prototype memory at
      every task, so when task t+1 arrives it becomes a negative for every
      expert created before it. An expert frozen after task j could never learn
      that on its own, which is exactly why rejection has to be a separate
      plastic module.

    `mode` selects how the heads are trained and compared:

    - "independent": each head is a separate binary classifier with balanced
      BCE (pos_weight = n_neg / n_pos). E0 measured that this is NOT mutually
      calibrated: macro AUROC is 0.93 (a within-head ranking) while the
      cross-head comparison that reranking needs - does the true expert beat
      the router's wrong pick - wins only 32.8% of the time.
    - "softmax": one joint cross-entropy over experts, so the scores are
      normalised against each other by construction. Same parameters, same
      expert-conditioning, but the heads can be compared.
    """

    def __init__(self, dim: int, mode: str = "softmax"):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.heads = nn.ModuleList()

    def __len__(self) -> int:
        return len(self.heads)

    def grow(self, device=None) -> None:
        head = AcceptHead(self.dim)
        if device is not None:
            head = head.to(device)
        elif len(self.heads) > 0:
            head = head.to(self.heads[0].weight.device)
        self.heads.append(head)

    def logits(self, zs: list[torch.Tensor]) -> torch.Tensor:
        """[B, N] logits; `zs[j]` is the input for expert j's head.

        Taking a per-expert input list (rather than one shared tensor) is what
        lets a head read its own expert's adapted representation
        `z + A_j(z)` instead of the shared frozen space.
        """
        if not self.heads:
            return torch.empty(zs[0].size(0), 0, device=zs[0].device)
        return torch.cat([head(z_j) for head, z_j in zip(self.heads, zs)], dim=-1)

    def scores(self, zs: list[torch.Tensor]) -> torch.Tensor:
        """[B, N] acceptance scores (probabilities in both modes)."""
        logits = self.logits(zs)
        if logits.numel() == 0:
            return logits
        return F.softmax(logits, dim=-1) if self.mode == "softmax" else logits.sigmoid()

    def fit(
        self,
        zs: list[torch.Tensor],
        task_of_row: torch.Tensor,
        steps: int = 300,
        lr: float = 0.01,
    ) -> float:
        """Fit on the prototype memory, each head on its own expert's inputs.

        At task 0 there are no negatives, so the bank is left at its
        conservative init.
        """
        n_experts = len(self.heads)
        if n_experts < 2:
            return float("nan")
        params = [p for head in self.heads for p in head.parameters()]
        opt = torch.optim.Adam(params, lr=lr)
        last = 0.0
        for _ in range(steps):
            opt.zero_grad()
            logits = self.logits(zs)
            if self.mode == "softmax":
                loss = F.cross_entropy(logits, task_of_row)
            else:
                loss = torch.zeros((), device=zs[0].device)
                for j in range(n_experts):
                    y = (task_of_row == j).float()
                    n_pos = float(y.sum())
                    n_neg = float(y.numel() - y.sum())
                    if n_pos == 0 or n_neg == 0:
                        continue
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logits[:, j], y, pos_weight=torch.tensor(n_neg / n_pos)
                    )
                loss = loss / max(n_experts, 1)
            loss.backward()
            opt.step()
            last = float(loss.item())
        return last


def auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney AUROC, no sklearn dependency."""
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    pos = pos.detach().float().cpu()
    neg = neg.detach().float().cpu()
    scores = torch.cat([pos, neg])
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    n_pos = float(pos.numel())
    n_neg = float(neg.numel())
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fpr_at_tpr(pos: torch.Tensor, neg: torch.Tensor, tpr: float = 0.95) -> float:
    """FPR at the threshold that reaches the requested TPR (0 when impossible)."""
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    pos = pos.detach().float().cpu()
    neg = neg.detach().float().cpu()
    threshold = torch.quantile(pos, 1.0 - tpr)
    return float((neg >= threshold).float().mean())


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tasks(cache_path: str):
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = payload["meta"]
    tasks = []
    for row in payload["tasks"]:
        splits = {}
        for name in ("train", "val", "test"):
            feats, labels = row["splits"][name]
            splits[name] = (feats.float(), labels.long())
        tasks.append(
            {
                "task_id": int(row["task_id"]),
                "classes": [int(c) for c in row["classes"]],
                "splits": splits,
            }
        )
    return meta, tasks


def iter_batches(feats, labels, batch_size, generator=None):
    n = feats.size(0)
    perm = torch.randperm(n, generator=generator)
    for start in range(0, n - batch_size + 1, batch_size):
        idx = perm[start : start + batch_size]
        yield feats[idx], labels[idx]


class E0Runner:
    """Shared state: frozen features, growing head, class means, prototypes.

    `readout` selects the classifier (spec E0 Level comparison):
      "head"  trained cosine classifier over `W` (Level 2)
      "ncm"   nearest-class-mean over the frozen-space class means, zero
              classifier parameters and zero classifier forgetting (Level 1)
    """

    def __init__(self, tasks, device, scale=10.0, masked=True, readout="head"):
        self.tasks = tasks
        self.device = device
        self.dim = tasks[0]["splits"]["train"][0].size(1)
        self.num_classes = sum(len(t["classes"]) for t in tasks)
        self.masked = masked
        self.readout = readout
        self.head = CosineHead(self.dim, self.num_classes, scale).to(device)
        self.seen: list[int] = []
        # Per-class feature means in the FROZEN space: NCM read-out + prototype
        # router (nearest mean selects the adapter's task). With `readout="ncm"`
        # these ARE the classifier, so they are immutable class anchors and
        # never move once a class has been seen.
        self.means = torch.zeros(self.num_classes, self.dim)
        self.mean_task = torch.full((self.num_classes,), -1, dtype=torch.long)
        self.classes_of_task: dict[int, list[int]] = {}
        self.rerank_alpha = 1.0
        self.lambda_reject = 0.0
        self.rejector = RejectorBank(self.dim, mode="softmax").to(device)
        self.rejector_space = "shared"
        self.shared_adapter = False
        self.reject_tau = 0.5
        self.adapters: list[ResidualAdapter] = []
        # L_func anchors: prototype features + the model output at registration.
        self.proto_z: list[torch.Tensor] = []
        self.proto_p: list[torch.Tensor] = []
        self._proto_classes: list[int] = []

    # ---------- memory ----------

    @torch.no_grad()
    def register_means(self, task) -> None:
        """Frozen-space class means: NCM read-out and the prototype router."""
        feats, labels = task["splits"]["train"]
        feats = feats.to(self.device)
        labels = labels.to(self.device)
        self.classes_of_task[task["task_id"]] = list(task["classes"])
        for c in task["classes"]:
            mask = labels == c
            if not bool(mask.any()):
                continue
            self.means[c] = feats[mask].mean(dim=0).cpu()
            self.mean_task[c] = task["task_id"]

    @torch.no_grad()
    def register_anchors(self, task) -> None:
        """Prototype anchors: the model's output on each class mean, with the
        class's own task adapter applied (the same semantics as v1's o_p)."""
        for c in task["classes"]:
            z = self.means[c].to(self.device)
            self.proto_z.append(z.detach().clone())
            self._proto_classes.append(int(c))
            if self.adapters and self.shared_adapter:
                z_out = self.apply_adapters(z.unsqueeze(0), None).squeeze(0)
            elif self.adapters and task["task_id"] < len(self.adapters):
                tids = torch.full(
                    (1,), task["task_id"], dtype=torch.long, device=self.device
                )
                z_out = self.apply_adapters(z.unsqueeze(0), tids).squeeze(0)
            else:
                z_out = z
            self.proto_p.append(self.predict_probs(z_out.unsqueeze(0)).squeeze(0))

    # ---------- forward ----------

    def apply_adapters(self, z: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        if not self.adapters:
            return z
        if self.shared_adapter:
            # L2 rung: one adapter for every task, so routing is irrelevant.
            return self.adapters[0].transform(z)
        out = z.clone()
        for t in torch.unique(task_ids).tolist():
            mask = task_ids == t
            out[mask] = self.adapters[int(t)].transform(z[mask])
        return out

    def _mask_unseen(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.masked:
            return logits
        unseen = torch.ones(self.num_classes, dtype=torch.bool, device=logits.device)
        unseen[self.seen] = False
        return logits.masked_fill(unseen, -1e9)

    def route(self, z: torch.Tensor, oracle_tasks=None) -> torch.Tensor:
        """Prototype router: nearest seen class mean -> its task id."""
        if oracle_tasks is not None:
            return oracle_tasks
        seen = torch.tensor(self.seen, dtype=torch.long)
        means = self.means[seen].to(z.device)
        sim = F.normalize(z, dim=-1) @ F.normalize(means, dim=-1).t()
        return self.mean_task[seen].to(z.device)[sim.argmax(dim=-1)]

    def route_topk(self, z: torch.Tensor, k: int, temp=None):
        """Top-k tasks by best class-mean similarity, with mixture weights.

        Task score = max cosine similarity over that task's seen classes. The
        weights are uniform by default; `temp` switches to a softmax over the
        k scores. topk=1 reproduces the hard prototype router exactly.
        """
        seen = torch.tensor(self.seen, dtype=torch.long)
        means = self.means[seen].to(z.device)
        tasks = self.mean_task[seen].to(z.device)
        sim = F.normalize(z, dim=-1) @ F.normalize(means, dim=-1).t()
        n_tasks = int(tasks.max().item()) + 1
        task_sim = torch.full((z.size(0), n_tasks), -2.0, device=z.device)
        task_sim.scatter_reduce_(
            1, tasks.unsqueeze(0).expand(z.size(0), -1), sim, reduce="amax"
        )
        top = task_sim.topk(min(k, n_tasks), dim=-1)
        if temp is None:
            weights = torch.full_like(top.values, 1.0 / top.values.size(1))
        else:
            weights = F.softmax(top.values / temp, dim=-1)
        return top.indices, weights, task_sim

    def forward_rerank(self, z: torch.Tensor, k: int, alpha: float) -> torch.Tensor:
        """Factorised score: router prior over tasks + expert score for candidates.

            score_c = alpha * task_score(task(c)) + expert_score_c   if task(c) in top-k
            score_c = alpha * task_score(task(c))                    otherwise

        `alpha = 0` is pure expert filtering, `alpha -> inf` is the pure router.
        The point is that the candidate set is never hard: a non-candidate class
        can still win through its router prior, so coverage failures fall back
        gracefully instead of becoming guaranteed errors.
        """
        ids, _, task_sim = self.route_topk(z, k)
        prior = task_sim[:, self.mean_task.to(z.device).clamp(min=0)]
        logits = alpha * self.head.scale * prior
        for j in range(ids.size(1)):
            for t in ids[:, j].unique().tolist():
                t = int(t)
                if t not in self.classes_of_task or t >= len(self.adapters):
                    continue
                rows = (ids[:, j] == t).nonzero(as_tuple=True)[0]
                z_mix = self.adapters[t].transform(z[rows])
                cols = torch.tensor(
                    self.classes_of_task[t], device=z.device, dtype=torch.long
                )
                logits[rows.unsqueeze(1), cols.unsqueeze(0)] += self._logits(z_mix)[
                    :, cols
                ]
        return self._mask_unseen(logits)

    def apply_adapters_mixture(
        self, z: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """z + sum_j w_j A_{t_j}(z) over the candidate tasks t_j (per sample)."""
        if not self.adapters:
            return z
        out = z.clone()
        for j in range(ids.size(1)):
            for t in ids[:, j].unique().tolist():
                mask = ids[:, j] == t
                delta = self.adapters[int(t)].transform(z[mask]) - z[mask]
                out[mask] = out[mask] + weights[mask, j].unsqueeze(-1) * delta
        return out

    def forward_topk_own(self, z: torch.Tensor, k: int, temp=None) -> torch.Tensor:
        """Expert filtering: score each class with ITS OWN task's adapter.

        For the k candidate tasks from the prototype router, every class of a
        candidate task is scored under that task's adapter; everything else is
        -inf. This avoids the representation-space dilution of a uniform
        mixture (which is out of distribution for the head) and matches how the
        adapters were trained: one adapter at a time. `k=1` reduces to
        route-then-classify-within-the-routed-task.
        """
        ids, _, _ = self.route_topk(z, k, temp)
        logits = torch.full(
            (z.size(0), self.num_classes), -1e9, device=z.device, dtype=z.dtype
        )
        for j in range(ids.size(1)):
            for t in ids[:, j].unique().tolist():
                t = int(t)
                if t not in self.classes_of_task or t >= len(self.adapters):
                    continue
                rows = (ids[:, j] == t).nonzero(as_tuple=True)[0]
                z_mix = self.adapters[t].transform(z[rows])
                cols = torch.tensor(
                    self.classes_of_task[t], device=z.device, dtype=torch.long
                )
                sub = self._logits(z_mix)
                logits[rows.unsqueeze(1), cols.unsqueeze(0)] = sub[:, cols]
        return logits

    def _logits(self, z_mix: torch.Tensor) -> torch.Tensor:
        if self.readout == "ncm":
            # Class means ARE the classifier: immutable anchors in the frozen
            # space, the adapter only moves the query. Zero classifier
            # parameters, so classifier forgetting is zero by construction.
            means = self.means.to(z_mix.device)
            sim = F.normalize(z_mix, dim=-1) @ F.normalize(means, dim=-1).t()
            return self.head.scale * sim
        return self.head(z_mix)

    def forward(
        self,
        z: torch.Tensor,
        adapters: bool = True,
        oracle=None,
        topk: int = 1,
        temp=None,
        mix_mode: str = "uniform",
    ):
        if not adapters:
            return self._mask_unseen(self._logits(z))
        if oracle is not None:
            z_mix = self.apply_adapters(z, oracle)
            return self._mask_unseen(self._logits(z_mix))
        if topk == 1 and mix_mode == "uniform":
            z_mix = self.apply_adapters(z, self.route(z))
            return self._mask_unseen(self._logits(z_mix))
        if mix_mode == "own":
            return self.forward_topk_own(z, topk, temp)
        if mix_mode == "rerank":
            return self.forward_rerank(z, topk, self.rerank_alpha)
        if mix_mode == "reject":
            return self.forward_reject(z, topk, self.lambda_reject)
        ids, weights, _ = self.route_topk(z, topk, temp)
        z_mix = self.apply_adapters_mixture(z, ids, weights)
        return self._mask_unseen(self._logits(z_mix))

    @torch.no_grad()
    def predict_probs(self, z: torch.Tensor) -> torch.Tensor:
        """Registered output distribution: masked and renormalized over seen classes."""
        return F.softmax(self._mask_unseen(self._logits(z)), dim=-1)

    def l_func(self) -> torch.Tensor:
        """Anchor the registered output distribution on the stored prototypes.

        The anchor must be evaluated through the SAME adapter path the model
        would use on those prototypes, otherwise the constraint is applied to a
        different representation than the one it is supposed to protect:

        - per-task adapters: each prototype goes through its own task's adapter
          (which is what `register_anchors` recorded and what routing does at
          eval time);
        - shared adapter: every prototype goes through it, so the constraint is
          a real one instead of a no-op;
        - no adapters: the frozen feature is the representation.
        """
        if not self.proto_z:
            return torch.zeros((), device=self.device)
        z = torch.stack(self.proto_z).to(self.device)
        p = torch.stack(self.proto_p).to(self.device)
        if self.adapters:
            if self.shared_adapter:
                z_in = self.apply_adapters(z, None)
            else:
                tids = torch.tensor(
                    [int(self.mean_task[int(c)]) for c in self._proto_classes],
                    device=self.device,
                )
                z_in = self.apply_adapters(z, tids)
        else:
            z_in = z
        return F.cross_entropy(self._mask_unseen(self._logits(z_in)), p)

    # ---------- evaluation ----------

    @torch.no_grad()
    def filtering_stats(self, k: int, temp=None) -> dict:
        """Coverage vs conditional accuracy of expert filtering.

        `coverage` = fraction of test samples whose true task is among the
        router's top-k candidates. `acc_covered` = accuracy of the own-adapter
        scoring restricted to those samples. If `acc_covered` is high but
        `acc_overall` is low, the scoring works and the candidates are fine, so
        the loss comes from spurious competitors among the candidates; if
        `acc_covered` is also low, the experts cannot reject other tasks' data.
        """
        tot = corr = cov_n = cov_corr = 0
        for task in self.tasks:
            feats, labels = task["splits"]["test"]
            feats, labels = feats.to(self.device), labels.to(self.device)
            ids, _, _ = self.route_topk(feats, k, temp)
            covered = (ids == task["task_id"]).any(dim=-1)
            logits = self.forward_topk_own(feats, k, temp)
            correct = logits.argmax(dim=-1) == labels
            tot += feats.size(0)
            corr += int(correct.sum())
            cov_n += int(covered.sum())
            cov_corr += int(correct[covered].sum())
        return {
            "coverage": cov_n / max(tot, 1),
            "acc_overall": corr / max(tot, 1),
            "acc_covered": cov_corr / max(cov_n, 1),
        }

    def rejector_inputs(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Per-expert input for the acceptance heads.

        `space="shared"` gives every head the frozen `z` (the E0 MVP that
        failed: its errors correlated with the router's because both read the
        same space). `space="local"` gives head j its own expert's adapted
        representation `z + A_j(z)`, which carries expert-conditioned evidence
        the router does not have. Full-N, as in the shared case: restricting to
        K candidates would change the inference candidate policy at the same
        time as the mechanism.
        """
        if self.rejector_space == "local" and self.adapters:
            return [z + adapter(z) for adapter in self.adapters]
        return [z] * len(self.rejector)

    def fit_rejector(self, steps: int = 300, lr: float = 0.01) -> float:
        """Re-fit every acceptance head on the full prototype memory."""
        if len(self.rejector) == 0:
            return float("nan")
        with torch.no_grad():
            z = torch.stack(self.proto_z).to(self.device)
            task_of_row = torch.tensor(
                [int(self.mean_task[int(c)]) for c in self._proto_classes],
                device=self.device,
            )
            zs = self.rejector_inputs(z)
        return self.rejector.fit(zs, task_of_row, steps=steps, lr=lr)

    def forward_reject(self, z: torch.Tensor, k: int, lam: float) -> torch.Tensor:
        """Acceptance reranking of the HARD ROUTE in a single common space.

            base_c  = logit_c under the top-1 expert's adapter   (all classes)
            score_c = base_c + lam * log r_{task(c)}(z)          (all classes)

        Why this shape, and not per-candidate adapters: scoring each class under
        its own task's adapter makes the comparison across candidates
        inconsistent (the same class gets different scores depending on the
        candidate set), and with lam > 0 it penalises the top-1 task's own
        classes, which is where the correct answer usually is. Here the base is
        one fixed readout, so `lam = 0` reproduces hard top-1 routing EXACTLY
        and any change is attributable to acceptance alone.

        The reranking is applied to every class, not only the K candidates: the
        rejector is a single linear map producing all N scores at once
        (O(Nd) ~= 15K MACs at d=768, N=20), so restricting it would save
        nothing and would make the score scale depend on the candidate set.
        The penalty is one-sided (log r <= 0): a confidently accepted expert is
        untouched, a rejected one is demoted, and a class whose expert is
        rejected cannot be promoted above a better-accepted one.
        """
        base = self._mask_unseen(self._logits(self.apply_adapters(z, self.route(z))))
        if lam == 0.0:
            return base
        log_r = (
            self.rejector.scores(self.rejector_inputs(z)).clamp(min=1e-6).log()
        )  # [B, N]
        task_of_class = self.mean_task.to(z.device).clamp(min=0)  # [C]
        return base + lam * log_r[:, task_of_class]

    def rejector_auroc(self) -> dict:
        """Threshold-free acceptance quality.

        Two different questions, deliberately separated:

        - `auroc_macro`: within-head ranking (expert j's own task vs all
          others). This is what a standard OOD report shows.
        - `cross_auc`: the cross-head ranking that reranking actually needs -
          for every ordered pair (i, j), does expert i's score beat expert j's
          on expert i's own data, averaged over pairs. A perfectly calibrated
          rejector has cross_auc = 1; independent sigmoid heads have no reason
          to.
        """
        if len(self.rejector) == 0:
            return {}
        per_expert = []
        all_z, all_task = [], []
        for task in self.tasks:
            feats, _ = task["splits"]["test"]
            all_z.append(feats.to(self.device))
            all_task.append(
                torch.full((feats.size(0),), task["task_id"], device=self.device)
            )
        z = torch.cat(all_z)
        task_of = torch.cat(all_task)
        with torch.no_grad():
            scores = self.rejector.scores(self.rejector_inputs(z)).cpu()
        task_of = task_of.cpu()
        n_experts = scores.size(1)
        for j in range(n_experts):
            s_j = scores[:, j]
            per_expert.append(
                {
                    "auroc": auroc(s_j[task_of == j], s_j[task_of != j]),
                    "fpr95": fpr_at_tpr(s_j[task_of == j], s_j[task_of != j]),
                }
            )
        pair_aucs = []
        for i in range(n_experts):
            own = scores[task_of == i]
            if own.size(0) == 0:
                continue
            others = own.clone()
            others[:, i] = -1.0
            pair_aucs.append(float((own[:, i].unsqueeze(1) > others).float().mean()))
        return {
            "auroc_macro": float(np.nanmean([e["auroc"] for e in per_expert])),
            "fpr95_macro": float(np.nanmean([e["fpr95"] for e in per_expert])),
            "cross_auc": float(np.mean(pair_aucs)) if pair_aucs else float("nan"),
            "per_expert": per_expert,
        }

    @torch.no_grad()
    def attribution(self, k: int, lam: float) -> dict:
        """Where does the final prediction come from?

        Decomposes the pipeline into candidate recall, acceptance precision and
        class accuracy, so a gain can be attributed to a mechanism instead of
        to the whole system.
        """
        tot = 0
        correct = 0
        top1_ok = topk_ok = 0
        correct_top1_ok = correct_top1_bad = 0
        accept_tp = accept_fp = accept_fn = 0
        pair_win = pair_n = 0
        for task in self.tasks:
            feats, labels = task["splits"]["test"]
            feats, labels = feats.to(self.device), labels.to(self.device)
            ids, _, _ = self.route_topk(feats, k)
            covered = (ids == task["task_id"]).any(dim=-1)
            logits = self.forward_reject(feats, k, lam)
            pred = logits.argmax(dim=-1)
            ok = pred == labels
            tot += feats.size(0)
            correct += int(ok.sum())
            top1_ok += int((ids[:, 0] == task["task_id"]).sum())
            topk_ok += int(covered.sum())
            correct_top1_ok += int(ok[ids[:, 0] == task["task_id"]].sum())
            correct_top1_bad += int(ok[ids[:, 0] != task["task_id"]].sum())
            # Acceptance precision/recall at tau, over candidate (sample, expert)
            # pairs: "accepted" = r above tau, "correct" = that expert is the
            # sample's true task.
            r = self.rejector.scores(self.rejector_inputs(feats))
            for j in range(k):
                cand = ids[:, j]
                accepted = r.gather(1, cand.unsqueeze(1)).squeeze(1) >= self.reject_tau
                is_true = cand == task["task_id"]
                accept_tp += int((accepted & is_true).sum())
                accept_fp += int((accepted & ~is_true).sum())
                accept_fn += int((~accepted & is_true).sum())
            # The single condition under which acceptance reranking can help:
            # when the router's top-1 is wrong, does the true expert score
            # higher than the router's pick? Below 0.5 the reranking is
            # guaranteed to demote correct answers more often than it rescues
            # wrong ones.
            wrong = ids[:, 0] != task["task_id"]
            if bool(wrong.any()):
                true_t = torch.full(
                    (feats.size(0),),
                    task["task_id"],
                    device=feats.device,
                    dtype=torch.long,
                )
                r_true = r.gather(1, true_t.unsqueeze(1)).squeeze(1)
                r_top1 = r.gather(1, ids[:, 0].unsqueeze(1)).squeeze(1)
                pair_win += int((r_true[wrong] > r_top1[wrong]).sum())
                pair_n += int(wrong.sum())
        prec = accept_tp / max(accept_tp + accept_fp, 1)
        rec = accept_tp / max(accept_tp + accept_fn, 1)
        return {
            "recall_at_k": topk_ok / max(tot, 1),
            "recall_at_1": top1_ok / max(tot, 1),
            "accept_precision": prec,
            "accept_recall": rec,
            "accept_f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "pairwise_win_rate_when_top1_wrong": pair_win / max(pair_n, 1),
            "acc_overall": correct / max(tot, 1),
            "acc_given_top1_correct": correct_top1_ok / max(top1_ok, 1),
            "acc_given_top1_wrong": correct_top1_bad / max(tot - top1_ok, 1),
        }

    @torch.no_grad()
    def evaluate_task(
        self, task, adapters=True, oracle=None, topk=1, temp=None, mix_mode="uniform"
    ) -> float:
        feats, labels = task["splits"]["test"]
        feats, labels = feats.to(self.device), labels.to(self.device)
        correct = 0
        for start in range(0, feats.size(0), 512):
            z = feats[start : start + 512]
            y = labels[start : start + 512]
            logits = self.forward(
                z,
                adapters=adapters,
                oracle=oracle,
                topk=topk,
                temp=temp,
                mix_mode=mix_mode,
            )
            correct += int((logits.argmax(dim=-1) == y).sum())
        return correct / max(feats.size(0), 1)


def routing_recall(runner: E0Runner, ks=(1, 2, 3, 5)) -> dict:
    """Top-K task recall of the prototype router on the test splits.

    "Is the true task among the K nearest class means?" This is the ceiling of
    any hard or soft task-routing scheme that reads the FROZEN space, and it is
    the number that decides whether the expert-adaptation ceiling (oracle
    routing) is reachable without a task id.
    """
    runner.seen = sorted({c for t in runner.tasks for c in t["classes"]})
    seen = torch.tensor(runner.seen, dtype=torch.long)
    means = runner.means[seen].to(runner.device)
    task_of = runner.mean_task[seen].to(runner.device)
    hits = {k: 0 for k in ks}
    total = 0
    with torch.no_grad():
        for task in runner.tasks:
            feats, _ = task["splits"]["test"]
            feats = feats.to(runner.device)
            sim = F.normalize(feats, dim=-1) @ F.normalize(means, dim=-1).t()
            topk_tasks = task_of[sim.topk(max(ks), dim=-1).indices]
            for k in ks:
                hits[k] += int((topk_tasks[:, :k] == task["task_id"]).any(dim=-1).sum())
            total += feats.size(0)
    return {f"task_recall@{k}": hits[k] / max(total, 1) for k in ks}


def run_ncm(runner: E0Runner) -> dict:
    """Zero-training NCM: cosine similarity to the class means."""
    R = np.zeros((len(runner.tasks), len(runner.tasks)), dtype=np.float32)
    for t, task in enumerate(runner.tasks):
        runner.register_means(task)
        runner.seen.extend(task["classes"])
        for i in range(t + 1):
            feats, labels = runner.tasks[i]["splits"]["test"]
            feats, labels = feats.to(runner.device), labels.to(runner.device)
            seen = torch.tensor(runner.seen, dtype=torch.long)
            means = runner.means[seen].to(runner.device)
            preds_all = []
            for start in range(0, feats.size(0), 512):
                z = feats[start : start + 512]
                sim = F.normalize(z, dim=-1) @ F.normalize(means, dim=-1).t()
                preds_all.append(seen.to(sim.device)[sim.argmax(dim=-1)].cpu())
            preds = torch.cat(preds_all)
            R[t, i] = float((preds == labels.cpu()).float().mean())
    out = summarize(
        R, extra={"params": 0, "memory_bytes": int(runner.num_classes * runner.dim * 4)}
    )
    out.update(routing_recall(runner))
    return out


def run_joint(runner: E0Runner, args) -> dict:
    """Upper bound: one classifier trained on every task's data jointly."""
    all_feats = torch.cat([t["splits"]["train"][0] for t in runner.tasks], dim=0)
    all_labels = torch.cat([t["splits"]["train"][1] for t in runner.tasks], dim=0)
    runner.seen = sorted({c for t in runner.tasks for c in t["classes"]})
    opt = torch.optim.Adam(runner.head.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    for _ in range(args.epochs):
        for z, y in iter_batches(all_feats, all_labels, args.batch_size, gen):
            z, y = z.to(runner.device), y.to(runner.device)
            loss = F.cross_entropy(runner.forward(z, adapters=False), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    R = np.zeros((len(runner.tasks), len(runner.tasks)), dtype=np.float32)
    for i in range(len(runner.tasks)):
        acc = runner.evaluate_task(runner.tasks[i], adapters=False)
        R[-1, i] = acc
    return summarize(
        R,
        extra={
            "params": int(sum(p.numel() for p in runner.head.parameters())),
            "memory_bytes": int(runner.num_classes * runner.dim * 4),
        },
    )


def run_class_incremental(
    runner: E0Runner, args, use_adapters: bool, use_func: bool = True
) -> dict:
    """Class-incremental: new rows train, old rows freeze, optional adapters.

    Readout is `runner.readout`:
      "head" - new head rows train (old rows are gradient-masked), Level 2.
      "ncm"  - the classifier is the immutable class means; only the adapter
               trains (Level 1). Class means must exist before training because
               they *are* the readout.

    `use_func=False` disables L_func, which is the pure growing-head probe used
    as the M4 red line: with `rank=0` and `lambda_func=0` the adapter path must
    reproduce it exactly (identical logits, no RNG consumed for a zero-element
    parameter).
    """
    ncm_readout = runner.readout == "ncm"
    n = len(runner.tasks)
    eval_ks = sorted({1, *[int(k) for k in args.eval_topk.split(",") if k]})
    R_by_k = {k: np.zeros((n, n), dtype=np.float32) for k in eval_ks}
    R_oracle = np.zeros((n, n), dtype=np.float32)
    adapter_params = 0
    for t, task in enumerate(runner.tasks):
        new_rows = task["classes"]
        runner.seen = sorted(set(runner.seen) | set(new_rows))
        if ncm_readout:
            # The class means are the classifier, so they must be registered
            # before the adapter trains. Immutable anchors: nothing to forget.
            runner.register_means(task)

        # Old head rows stay frozen: their gradients are masked in the backward
        # hook below, which is the only correct way to freeze a *subset* of one
        # parameter tensor (indexing W[c] returns a copy, so an optimizer over
        # those views would silently train nothing).
        row_mask = torch.zeros(runner.num_classes, dtype=torch.bool)
        row_mask[new_rows] = True

        def _mask_rows(grad, mask=row_mask):
            if grad is None:
                return grad
            grad = grad.clone()
            grad[~mask.to(grad.device)] = 0.0
            return grad

        handle = runner.head.W.register_hook(_mask_rows)
        trainable = []
        if not ncm_readout:
            trainable.append(runner.head.W)
        if use_adapters:
            if runner.shared_adapter:
                # L2 rung: create the single adapter once, keep training it.
                if not runner.adapters:
                    runner.adapters.append(
                        ResidualAdapter(runner.dim, args.rank).to(runner.device)
                    )
                    adapter_params += sum(
                        p.numel() for p in runner.adapters[0].parameters()
                    )
                trainable.extend(runner.adapters[0].parameters())
            else:
                adapter = ResidualAdapter(runner.dim, args.rank).to(runner.device)
                runner.adapters.append(adapter)
                adapter_params += sum(p.numel() for p in adapter.parameters())
                trainable.extend(adapter.parameters())

        if trainable:
            opt = torch.optim.Adam(trainable, lr=args.lr)
            feats, labels = task["splits"]["train"]
            gen = torch.Generator().manual_seed(args.seed + t)
            for _ in range(args.epochs):
                for z, y in iter_batches(feats, labels, args.batch_size, gen):
                    z, y = z.to(runner.device), y.to(runner.device)
                    task_ids = torch.full(
                        (z.size(0),), t, dtype=torch.long, device=runner.device
                    )
                    z_mix = runner.apply_adapters(z, task_ids) if use_adapters else z
                    logits = runner._mask_unseen(runner._logits(z_mix))
                    loss = F.cross_entropy(logits, y)
                    if use_func and args.lambda_func > 0:
                        loss = loss + args.lambda_func * runner.l_func()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        handle.remove()

        if not ncm_readout:
            runner.register_means(task)
        runner.register_anchors(task)
        if args.use_rejector:
            # Grow the acceptance bank with the new expert, then re-fit every
            # head on the full prototype memory: this is where task t becomes a
            # negative for all experts created before it. Anchors must already
            # be registered so the new expert has positives to fit against.
            runner.rejector.grow(runner.device)
            runner.fit_rejector(steps=args.reject_steps)
        for i in range(t + 1):
            for k in eval_ks:
                R_by_k[k][t, i] = runner.evaluate_task(
                    runner.tasks[i],
                    adapters=use_adapters,
                    topk=k,
                    temp=args.mix_temp,
                    mix_mode=args.mix_mode,
                )
        if use_adapters and args.oracle_routing:
            for i in range(t + 1):
                oracle = torch.full(
                    (runner.tasks[i]["splits"]["test"][0].size(0),),
                    i,
                    dtype=torch.long,
                    device=runner.device,
                )
                R_oracle[t, i] = runner.evaluate_task(
                    runner.tasks[i], adapters=True, oracle=oracle
                )
    head_params = (
        0 if ncm_readout else int(sum(p.numel() for p in runner.head.parameters()))
    )
    out = summarize(
        R_by_k[1],
        extra={
            "readout": runner.readout,
            "params": head_params + adapter_params,
            "trainable_params": adapter_params + (0 if ncm_readout else 5 * runner.dim),
            "memory_bytes": int(
                runner.num_classes * runner.dim * 4
                + sum(p.numel() * 4 for p in runner.proto_p)
            ),
            "rank": args.rank if use_adapters else 0,
            "mix_mode": args.mix_mode,
            "mix_temp": args.mix_temp,
        },
    )
    if len(eval_ks) > 1:
        out["topk"] = {
            k: {
                "avg_accuracy": summarize(R_by_k[k])["avg_accuracy"],
                "forgetting": summarize(R_by_k[k])["forgetting"],
            }
            for k in eval_ks
        }
    if use_adapters and args.oracle_routing:
        out["oracle_routing_avg_accuracy"] = float(
            np.mean(R_oracle[len(runner.tasks) - 1, :])
        )
    if use_adapters and args.mix_mode == "own":
        out["filtering"] = {
            k: runner.filtering_stats(k, args.mix_temp) for k in eval_ks if k > 1
        }
    if use_adapters and args.use_rejector:
        out["rejector"] = runner.rejector_auroc()
        out["attribution"] = {
            k: runner.attribution(k, args.lambda_reject) for k in eval_ks
        }
        out["rejector_mode"] = args.rejector_mode
        out["rejector_space"] = args.rejector_space
        out["lambda_reject"] = args.lambda_reject
        out["reject_tau"] = args.reject_tau
        out["oracle_gap_closure"] = {
            k: (
                (out["topk"][k]["avg_accuracy"] - out["avg_accuracy"])
                / max(
                    out.get("oracle_routing_avg_accuracy", float("nan"))
                    - out["avg_accuracy"],
                    1e-9,
                )
            )
            for k in out.get("topk", {})
        }
    return out


def run_shared_adapter(runner: E0Runner, args, joint: bool) -> dict:
    """L2 rung (rule 8): ONE adapter for every task, no expert bank, no routing.

    `joint=True` trains on all tasks at once: the upper bound for what a single
    adapter can add to the frozen features. `joint=False` trains sequentially
    with the same protection as the per-task runs (frozen old head rows plus
    L_func).

    This is the control that decides whether the expert bank is justified at
    all: if the per-task bank cannot beat one shared adapter, the bank is
    buying routing complexity rather than capacity.
    """
    runner.shared_adapter = True
    n = len(runner.tasks)
    if joint:
        all_feats = torch.cat([t["splits"]["train"][0] for t in runner.tasks], dim=0)
        all_labels = torch.cat([t["splits"]["train"][1] for t in runner.tasks], dim=0)
        runner.seen = sorted({c for t in runner.tasks for c in t["classes"]})
        for task in runner.tasks:
            runner.register_means(task)
            runner.register_anchors(task)
        adapter = ResidualAdapter(runner.dim, args.rank).to(runner.device)
        runner.adapters.append(adapter)
        trainable = list(adapter.parameters()) + [runner.head.W]
        opt = torch.optim.Adam(trainable, lr=args.lr)
        gen = torch.Generator().manual_seed(args.seed)
        for _ in range(args.epochs):
            for z, y in iter_batches(all_feats, all_labels, args.batch_size, gen):
                z, y = z.to(runner.device), y.to(runner.device)
                logits = runner._mask_unseen(
                    runner._logits(runner.apply_adapters(z, None))
                )
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        R = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            R[n - 1, i] = runner.evaluate_task(runner.tasks[i], adapters=True)
        return summarize(
            R,
            extra={
                "readout": runner.readout,
                "params": int(sum(p.numel() for p in adapter.parameters()))
                + int(sum(p.numel() for p in runner.head.parameters())),
                "trainable_params": int(sum(p.numel() for p in adapter.parameters()))
                + runner.num_classes * runner.dim,
                "memory_bytes": int(
                    runner.num_classes * runner.dim * 4
                    + sum(p.numel() * 4 for p in runner.proto_p)
                ),
                "rank": args.rank,
                "shared_adapter": "joint",
            },
        )
    out = run_class_incremental(runner, args, use_adapters=True, use_func=True)
    out["shared_adapter"] = "sequential"
    return out


def emit_contracts(results: dict, args, meta: dict, tasks) -> dict:
    """Emit a contract record per mode (docs/MEASUREMENT_CONTRACT.md, S0).

    This is the *native* production path: factors and metrics are stated
    explicitly instead of being inferred from a runner result dict, which is
    how a new experiment is supposed to enter the framework.
    """
    from pal_moe.evaluation.schema import build_run_record, satisfied_blocks

    class_order = [list(t["classes"]) for t in tasks]
    records = {}
    for key, m in results["modes"].items():
        if not isinstance(m, dict) or "avg_accuracy" not in m:
            continue
        rank = m.get("rank")
        records[key] = build_run_record(
            factors={
                "dataset": meta["dataset"],
                "protocol": "class_il",
                "task_id_at_inference": False,
                "num_tasks": len(tasks),
                "classes_per_task": len(tasks[0]["classes"]),
                "class_order": class_order,
                "task_order_seed": None,
                "seed": args.seed,
                "model_family": key,
                "backbone": meta["encoder_arch"],
                "backbone_pretraining": f"{meta['encoder_weights']}_frozen",
                "readout": m.get("readout", "head"),
                "expert": (
                    f"residual_adapter_r{rank}"
                    if rank
                    else ("residual_adapter_r0" if m.get("shared_adapter") else "none")
                ),
                "budget": {
                    "memory_bytes": None,
                    "params": None,
                    "compute_flops": None,
                    "steps": None,
                },
            },
            metrics={
                "learning": {
                    "accuracy": m["avg_accuracy"],
                    "forgetting": m["forgetting"],
                    "acc_matrix": m["R"],
                },
                "cost": {
                    "memory_bytes": m["memory_bytes"],
                    "stored_bytes": m["memory_bytes"],
                    "total_params": m["params"],
                    "trainable_params": m.get("trainable_params"),
                },
                "modular": {
                    "expert_count": len(m.get("R", [])) if "R" in m else None,
                    "task_recall_at_k": m.get("task_recall_at_k"),
                    "oracle_accuracy": m.get("oracle_routing_avg_accuracy"),
                },
            },
            provenance={
                "git_commit": meta.get("git_commit", "unknown"),
                "device": str(args.device),
                "command": "experiments/e0_representation_ceiling.py",
            },
        )
        records[key]["contract_blocks"] = satisfied_blocks(records[key])
    return records


def summarize(R: np.ndarray, extra: dict | None = None) -> dict:
    """Average accuracy + repo-convention forgetting (BENCHMARK.md)."""
    T = R.shape[0] - 1
    avg_acc = float(np.mean(R[T, : T + 1]))
    forget = []
    for i in range(T):
        forget.append(max(0.0, float(np.max(R[i : T + 1, i])) - float(R[T, i])))
    return {
        "avg_accuracy": avg_acc,
        "forgetting": float(np.mean(forget)) if forget else 0.0,
        "R": R.tolist(),
        **(extra or {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument(
        "--mode",
        default="all",
        choices=["ncm", "joint", "cl", "adapter", "adapter_ncm", "shared", "all"],
    )
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--ranks", default="0,8,32,64")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--oracle_routing", action="store_true")
    parser.add_argument(
        "--eval_topk",
        default="1",
        help="comma-separated top-k expert mixtures to evaluate (1 = hard route)",
    )
    parser.add_argument(
        "--mix_mode",
        default="uniform",
        choices=["uniform", "own", "rerank", "reject"],
        help="uniform = blend adapter outputs; own = hard expert filtering; rerank = router prior + expert score; reject = expert filtering with acceptance reranking and a hard-route fallback",
    )
    parser.add_argument("--use_rejector", action="store_true")
    parser.add_argument(
        "--rejector_space",
        default="shared",
        choices=["shared", "local"],
        help="shared = heads read frozen z (E0 MVP); local = head j reads z + A_j(z)",
    )
    parser.add_argument(
        "--lambda_reject",
        type=float,
        default=0.0,
        help="weight of log r in reject mode",
    )
    parser.add_argument("--reject_tau", type=float, default=0.5)
    parser.add_argument("--reject_steps", type=int, default=300)
    parser.add_argument(
        "--rejector_mode",
        default="softmax",
        choices=["softmax", "independent"],
        help="softmax = joint cross-entropy over experts (mutually calibrated); independent = per-expert balanced BCE",
    )
    parser.add_argument(
        "--rerank_alpha",
        type=float,
        default=1.0,
        help="weight of the router task prior in rerank mode (0 = pure filtering)",
    )
    parser.add_argument(
        "--mix_temp",
        type=float,
        default=None,
        help="softmax temperature for mixture weights; None = uniform over top-k",
    )
    parser.add_argument("--tag", default="", help="suffix for the output filename")
    parser.add_argument("--out", default="results/e0")
    args = parser.parse_args()

    device = torch.device(args.device)
    meta, tasks = load_tasks(args.cache)
    print(
        f"[E0] {meta['encoder_arch']} / {meta['encoder_weights']}, "
        f"{len(tasks)} tasks, dim={meta['feature_dim']}, device={device}"
    )

    results: dict = {"meta": meta, "args": vars(args), "modes": {}}
    t0 = time.time()

    if args.mode in ("ncm", "all"):
        set_seed(args.seed)
        runner = E0Runner(tasks, device, args.scale)
        results["modes"]["ncm"] = run_ncm(runner)
        print(
            f"[E0] ncm     acc={results['modes']['ncm']['avg_accuracy']:.4f} "
            f"F={results['modes']['ncm']['forgetting']:.4f}"
        )

    if args.mode in ("joint", "all"):
        set_seed(args.seed)
        runner = E0Runner(tasks, device, args.scale)
        results["modes"]["joint"] = run_joint(runner, args)
        print(
            f"[E0] joint   acc={results['modes']['joint']['avg_accuracy']:.4f} "
            f"F={results['modes']['joint']['forgetting']:.4f}"
        )

    if args.mode in ("cl", "all"):
        set_seed(args.seed)
        runner = E0Runner(tasks, device, args.scale)
        results["modes"]["cl"] = run_class_incremental(
            runner, args, use_adapters=False, use_func=False
        )
        print(
            f"[E0] cl      acc={results['modes']['cl']['avg_accuracy']:.4f} "
            f"F={results['modes']['cl']['forgetting']:.4f}"
        )
        # v2 objective without adapters: the same head protection, no capacity.
        set_seed(args.seed)
        runner = E0Runner(tasks, device, args.scale)
        results["modes"]["cl_func"] = run_class_incremental(
            runner, args, use_adapters=False, use_func=True
        )
        print(
            f"[E0] cl_func acc={results['modes']['cl_func']['avg_accuracy']:.4f} "
            f"F={results['modes']['cl_func']['forgetting']:.4f}"
        )

    if args.mode in ("adapter", "all"):
        ranks = (
            [args.rank]
            if args.mode == "adapter"
            else [int(r) for r in args.ranks.split(",")]
        )
        for rank in ranks:
            set_seed(args.seed)
            runner = E0Runner(tasks, device, args.scale)
            runner.rejector.mode = args.rejector_mode
            runner.rejector_space = args.rejector_space
            runner.rerank_alpha = args.rerank_alpha
            runner.lambda_reject = args.lambda_reject
            runner.reject_tau = args.reject_tau
            run_args = argparse.Namespace(**{**vars(args), "rank": rank})
            key = f"adapter_r{rank}"
            results["modes"][key] = run_class_incremental(
                runner, run_args, use_adapters=True, use_func=True
            )
            print(
                f"[E0] {key:11s} acc={results['modes'][key]['avg_accuracy']:.4f} "
                f"F={results['modes'][key]['forgetting']:.4f} "
                f"params={results['modes'][key]['params']}"
            )
        # M4 red line: rank 0 without L_func must equal the pure probe exactly.
        if "0" in [str(r) for r in ranks]:
            set_seed(args.seed)
            runner = E0Runner(tasks, device, args.scale)
            run_args = argparse.Namespace(
                **{**vars(args), "rank": 0, "lambda_func": 0.0}
            )
            results["modes"]["adapter_r0_nofunc"] = run_class_incremental(
                runner, run_args, use_adapters=True, use_func=False
            )
            delta = float(
                np.abs(
                    np.array(results["modes"]["adapter_r0_nofunc"]["R"])
                    - np.array(results["modes"]["cl"]["R"])
                ).max()
            )
            results["modes"]["adapter_r0_nofunc"]["red_line_max_R_delta"] = delta
            print(
                f"[E0] red line: adapter(r=0, no L_func) vs cl -> "
                f"max|dR|={delta:.2e} "
                f"(must be 0: {'PASS' if delta == 0.0 else 'FAIL'})"
            )

    # L2 rung (rule 8): one shared adapter for every task, joint and sequential.
    if args.mode in ("shared", "all"):
        for joint in (True, False):
            set_seed(args.seed)
            runner = E0Runner(tasks, device, args.scale)
            runner.rejector.mode = args.rejector_mode
            runner.rejector_space = args.rejector_space
            runner.rerank_alpha = args.rerank_alpha
            runner.lambda_reject = args.lambda_reject
            runner.reject_tau = args.reject_tau
            key = f"shared_{'joint' if joint else 'sequential'}"
            results["modes"][key] = run_shared_adapter(runner, args, joint=joint)
            print(
                f"[E0] {key:18s} acc={results['modes'][key]['avg_accuracy']:.4f} "
                f"F={results['modes'][key]['forgetting']:.4f} "
                f"params={results['modes'][key]['params']}"
            )

    # Level 1: adapter + NCM read-out (no classifier parameters at all).
    if args.mode in ("adapter_ncm", "all"):
        ncm_ranks = [int(r) for r in args.ranks.split(",")]
        for rank in ncm_ranks:
            set_seed(args.seed)
            runner = E0Runner(tasks, device, args.scale, readout="ncm")
            runner.rejector.mode = args.rejector_mode
            runner.rejector_space = args.rejector_space
            runner.rerank_alpha = args.rerank_alpha
            runner.lambda_reject = args.lambda_reject
            runner.reject_tau = args.reject_tau
            run_args = argparse.Namespace(**{**vars(args), "rank": rank})
            key = f"ncm_adapter_r{rank}"
            results["modes"][key] = run_class_incremental(
                runner, run_args, use_adapters=True, use_func=True
            )
            print(
                f"[E0] {key:15s} acc={results['modes'][key]['avg_accuracy']:.4f} "
                f"F={results['modes'][key]['forgetting']:.4f} "
                f"params={results['modes'][key]['params']}"
            )

    results["contracts"] = emit_contracts(results, args, meta, tasks)
    results["seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(
        args.out,
        f"e0_{args.mode}{args.tag}_{meta['encoder_arch']}_seed{args.seed}.json",
    )
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"[E0] wrote {path} ({results['seconds']}s)")


if __name__ == "__main__":
    main()
