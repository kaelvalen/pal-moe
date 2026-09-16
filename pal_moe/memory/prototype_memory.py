"""
Prototype Memory for Router and Expert Stability in PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts).

Maintains:
- v_p: Routing feature vectors (from frozen/EMA encoder)
- r_p: Past routing distributions g_old(v_p)
- o_p: Past expert output anchors E_old(h(x_p))
- x_p: Small exemplar feature set for drift verification and accuracy checks
- y_p: Exemplar ground-truth class labels for direct accuracy validation
- raw_x: Raw input samples to dynamically refresh representations upon encoder drift
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class Prototype:
    v_p: torch.Tensor  # [feature_dim]
    r_p: torch.Tensor  # [num_experts_at_creation]
    o_p: torch.Tensor  # [num_experts_at_creation, num_classes] (expert output anchors)
    task_id: int
    count: int = 1
    # Expert that was trained for this prototype's task (explicit task->expert
    # anchor, robust to router collapse). None = derive from r_p.
    owner_expert: Optional[int] = None
    x_p: Optional[torch.Tensor] = None  # [num_exemplars, feature_dim]
    y_p: Optional[torch.Tensor] = None  # [num_exemplars] labels of those exemplars
    # x_p/y_p serve two roles: (1) the joint-calibration exemplar batch
    # (features + labels) and (2) extra router-stability anchors (each exemplar
    # row is treated as a prototype centre whose r_p target is the prototype's).
    # They contain no raw inputs.
    raw_x: Optional[torch.Tensor] = (
        None  # [num_exemplars, input_dim] (raw inputs for refresh)
    )


class PrototypeMemory:
    """
    Manages prototype anchors to preserve both router decisions and expert outputs.
    Mitigates representation drift via EMA updates, bounded distance clustering,
    and direct capacity-control synchronization upon expert pruning/merging.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        distance_threshold: Optional[float] = 0.5,
        ema_alpha: float = 0.9,
        max_prototypes: int = 50,
        exemplars_per_proto: int = 5,
        store_raw: bool = False,
        max_prototypes_per_class: Optional[int] = None,
    ):
        self.feature_dim = feature_dim
        self.distance_threshold = distance_threshold
        self.ema_alpha = ema_alpha
        self.max_prototypes = max_prototypes
        self.exemplars_per_proto = exemplars_per_proto
        self.store_raw = store_raw
        # When set, eviction keeps a per-(task, class) budget instead of a
        # per-task one, preventing majority classes from crowding out the rest.
        self.max_prototypes_per_class = max_prototypes_per_class

        self.prototypes: List[Prototype] = []

        # Cached stacked tensors (prototype/route/output anchors). Invalidated on
        # every mutation so hot loops (stability losses) do not re-stack hundreds
        # of CPU tensors and re-transfer them to the device every batch.
        self._cache: Dict[Any, Any] = {}

    def __len__(self) -> int:
        return len(self.prototypes)

    def is_empty(self) -> bool:
        return len(self.prototypes) == 0

    def _invalidate_cache(self) -> None:
        if self._cache:
            self._cache.clear()

    def _get_cached(self, key: Any, builder: Any) -> Any:
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]

    def get_prototype_matrix(self, device: torch.device) -> Optional[torch.Tensor]:
        """Returns [P, feature_dim] matrix of all stored prototype vectors."""
        if self.is_empty():
            return None
        return self._get_cached(
            ("v", str(device)),
            lambda: torch.stack([p.v_p.to(device) for p in self.prototypes], dim=0),
        )

    def get_routing_matrix(self, num_experts: int, device: torch.device) -> torch.Tensor:
        """
        Returns [P, num_experts] historically padded routing distributions r_p
        (missing trailing columns filled with `pad`), aligned with
        `get_prototype_matrix`. Cached per (device, num_experts, pad).
        """

        def build() -> torch.Tensor:
            rows = []
            for p in self.prototypes:
                rp = p.r_p.to(device)
                n = rp.size(0)
                if n < num_experts:
                    pad = torch.full((num_experts - n,), 1e-4 / num_experts, device=device)
                    rp = torch.cat([rp, pad], dim=0)
                else:
                    rp = rp[:num_experts]
                    rp = rp / (rp.sum() + 1e-9)
                rows.append(rp)
            return torch.stack(rows, dim=0)

        return self._get_cached(("r", str(device), num_experts), build)

    def get_expert_anchor_matrix(
        self, num_experts: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Returns [P, num_experts] anchor distributions for prototype-anchored
        inference routing: one-hot for the prototype's explicit `owner_expert`
        when known, otherwise its historical `r_p` (padded). Cached until the
        memory mutates.
        """
        if self.is_empty():
            return None

        def build() -> torch.Tensor:
            rows = []
            for p in self.prototypes:
                if p.owner_expert is not None and p.owner_expert < num_experts:
                    row = torch.zeros(num_experts, device=device)
                    row[p.owner_expert] = 1.0
                else:
                    rp = p.r_p.to(device)
                    n = rp.size(0)
                    if n < num_experts:
                        pad = torch.full(
                            (num_experts - n,), 1e-4 / num_experts, device=device
                        )
                        row = torch.cat([rp, pad], dim=0)
                    else:
                        row = rp[:num_experts]
                        row = row / (row.sum() + 1e-9)
                rows.append(row)
            return torch.stack(rows, dim=0)

        return self._get_cached(("expert_anchor", str(device), num_experts), build)

    def get_output_matrix(
        self, num_experts: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Returns [P, num_experts, num_classes] zero-padded expert output anchors."""
        if self.is_empty():
            return None

        def build() -> torch.Tensor:
            num_classes = self.prototypes[0].o_p.size(-1)
            out = torch.zeros(
                len(self.prototypes), num_experts, num_classes, device=device
            )
            for i, p in enumerate(self.prototypes):
                n = min(p.o_p.size(0), num_experts)
                out[i, :n] = p.o_p[:n].to(device)
            return out

        return self._get_cached(("o", str(device), num_experts), build)

    def get_router_anchor_matrices(
        self, num_experts: int, device: torch.device
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns (V_anchor [M, D], R_targets [M, num_experts]) used by the router
        stability loss: each prototype centre plus its exemplar features, each
        mapped to its (padded) historical routing distribution.
        """
        if self.is_empty():
            return None

        def build() -> Tuple[torch.Tensor, torch.Tensor]:
            v_rows, r_rows = [], []
            for p in self.prototypes:
                rp = p.r_p.to(device)
                n = rp.size(0)
                if n < num_experts:
                    pad_size = num_experts - n
                    eps = 1e-4 / num_experts
                    rp_target = torch.cat(
                        [
                            rp * (1.0 - eps * pad_size),
                            torch.full((pad_size,), eps, device=device),
                        ],
                        dim=0,
                    )
                else:
                    rp_target = rp[:num_experts]
                    rp_target = rp_target / (rp_target.sum() + 1e-9)
                v_rows.append(p.v_p.to(device))
                r_rows.append(rp_target)
                if p.x_p is not None:
                    for i in range(p.x_p.size(0)):
                        v_rows.append(p.x_p[i].to(device))
                        r_rows.append(rp_target)
            return torch.stack(v_rows, dim=0), torch.stack(r_rows, dim=0)

        return self._get_cached(("anchor", str(device), num_experts), build)

    def get_raw_anchor(
        self, device: torch.device
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns (raw_rows [M, input_dim], owner_idx [M]) grouping every stored raw
        exemplar with the index of the prototype it belongs to (for the encoder
        stability loss). Cached until the memory mutates.
        """
        if self.is_empty():
            return None

        def build() -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
            rows, owners = [], []
            for p_idx, p in enumerate(self.prototypes):
                if p.raw_x is not None and p.raw_x.size(0) > 0:
                    rows.append(p.raw_x)
                    owners.append(torch.full((p.raw_x.size(0),), p_idx, dtype=torch.long))
            if not rows:
                return None
            return torch.cat(rows, dim=0).to(device), torch.cat(owners, dim=0).to(device)

        return self._get_cached(("raw", str(device)), build)

    def get_expert_anchors(
        self, expert_id: int, device: torch.device
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieves stored prototype representations and historical output anchors for a given expert.
        Returns:
            (v_mat, o_mat) where v_mat is [K, feature_dim] and o_mat is [K, num_classes] on device,
            or None if no prototypes contain anchors for expert_id.
        """
        if self.is_empty():
            return None
        v_list = []
        o_list = []
        for p in self.prototypes:
            if expert_id < p.o_p.size(0):
                v_list.append(p.v_p)
                o_list.append(p.o_p[expert_id])
        if not v_list:
            return None
        return torch.stack(v_list, dim=0).to(device), torch.stack(o_list, dim=0).to(
            device
        )

    def get_exemplar_batch(
        self, device: torch.device
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Gathers all stored exemplar feature vectors and labels across all prototypes.
        Returns:
            (all_feats, all_labels) on device, or None if no labeled exemplars exist.
        """
        feats = []
        labels = []
        for p in self.prototypes:
            if (
                p.x_p is not None
                and p.y_p is not None
                and p.x_p.size(0) > 0
                and p.y_p.size(0) > 0
            ):
                min_len = min(p.x_p.size(0), p.y_p.size(0))
                feats.append(p.x_p[:min_len])
                labels.append(p.y_p[:min_len])
        if not feats:
            return None
        return torch.cat(feats, dim=0).to(device), torch.cat(labels, dim=0).to(device)

    def compute_min_distance(self, x_feats: torch.Tensor) -> torch.Tensor:
        """
        Computes minimum Euclidean distance d(x, P) from each sample in x_feats to all prototypes.
        Args:
            x_feats: [batch_size, feature_dim]
        Returns:
            min_dist: [batch_size]
        """
        if self.is_empty():
            return torch.ones(x_feats.size(0), device=x_feats.device) * 2.0

        p_mat = self.get_prototype_matrix(x_feats.device)  # [P, D]
        dists = torch.cdist(x_feats, p_mat, p=2)
        min_dist, _ = torch.min(dists, dim=1)
        return min_dist

    def update_or_create_prototype(
        self,
        feat: torch.Tensor,
        routing_dist: torch.Tensor,
        expert_outputs: torch.Tensor,
        task_id: int,
        label: Optional[torch.Tensor] = None,
        raw_input: Optional[torch.Tensor] = None,
        owner_expert: Optional[int] = None,
    ) -> Prototype:
        """
        Processes a single feature vector:
        - If close to an existing prototype (<= distance_threshold), updates via EMA.
        - Otherwise, instantiates a new prototype.
        """
        proto, _ = self._update_or_create_with_matrix(
            feat,
            routing_dist,
            expert_outputs,
            task_id,
            label,
            raw_input,
            None,
            owner_expert=owner_expert,
        )
        return proto

    def _update_or_create_with_matrix(
        self,
        feat: torch.Tensor,
        routing_dist: torch.Tensor,
        expert_outputs: torch.Tensor,
        task_id: int,
        label: Optional[torch.Tensor],
        raw_input: Optional[torch.Tensor],
        work_matrix: Optional[torch.Tensor],
        owner_expert: Optional[int] = None,
    ) -> Tuple[Prototype, Optional[torch.Tensor]]:
        """
        Same as `update_or_create_prototype`, but reuses a caller-maintained
        [P, D] matrix of prototype centres so registering a batch does not
        re-stack every prototype for every sample. Returns the (possibly
        re-allocated) matrix alongside the touched prototype.
        """
        feat_detached = feat.detach().cpu()
        routing_detached = routing_dist.detach().cpu()
        expert_detached = expert_outputs.detach().cpu()
        y_detached = label.detach().cpu().view(1) if label is not None else None
        raw_detached = None
        if self.store_raw and raw_input is not None:
            r = raw_input.detach().cpu()
            if r.dim() == 1 or (r.dim() == 3 and r.size(0) in (1, 3)):
                raw_detached = r.unsqueeze(0)
            else:
                raw_detached = r

        if self.is_empty():
            new_proto = Prototype(
                v_p=feat_detached.clone(),
                r_p=routing_detached.clone(),
                o_p=expert_detached.clone(),
                task_id=task_id,
                owner_expert=owner_expert,
                count=1,
                x_p=feat_detached.unsqueeze(0).clone(),
                y_p=y_detached.clone() if y_detached is not None else None,
                raw_x=raw_detached.clone() if raw_detached is not None else None,
            )
            self.prototypes.append(new_proto)
            self._invalidate_cache()
            return new_proto, feat_detached.unsqueeze(0).clone()

        if work_matrix is None or work_matrix.size(0) != len(self.prototypes):
            work_matrix = torch.stack([p.v_p for p in self.prototypes], dim=0)

        # Find closest existing prototype against the caller-maintained matrix
        dists = torch.norm(work_matrix - feat_detached.unsqueeze(0), p=2, dim=1)
        min_dist, closest_idx = torch.min(dists, dim=0)
        closest = int(closest_idx.item())

        if min_dist.item() <= self.distance_threshold:
            # Update existing prototype with EMA
            proto = self.prototypes[closest]
            proto.v_p = (
                self.ema_alpha * proto.v_p + (1.0 - self.ema_alpha) * feat_detached
            )
            proto.count += 1
            if owner_expert is not None:
                proto.owner_expert = owner_expert
            work_matrix[closest] = proto.v_p
            # Maintain exemplars up to buffer size
            if proto.x_p is not None and proto.x_p.size(0) < self.exemplars_per_proto:
                proto.x_p = torch.cat([proto.x_p, feat_detached.unsqueeze(0)], dim=0)
                if y_detached is not None:
                    if proto.y_p is not None:
                        proto.y_p = torch.cat([proto.y_p, y_detached], dim=0)
                    else:
                        proto.y_p = y_detached.clone()
                if raw_detached is not None:
                    if proto.raw_x is not None:
                        proto.raw_x = torch.cat([proto.raw_x, raw_detached], dim=0)
                    else:
                        proto.raw_x = raw_detached.clone()
            self._invalidate_cache()
            return proto, work_matrix

        # Check capacity before creating new
        if len(self.prototypes) >= self.max_prototypes:
            self._prune_or_merge_least_used()
            work_matrix = torch.stack([p.v_p for p in self.prototypes], dim=0)

        new_proto = Prototype(
            v_p=feat_detached.clone(),
            r_p=routing_detached.clone(),
            o_p=expert_detached.clone(),
            task_id=task_id,
            owner_expert=owner_expert,
            count=1,
            x_p=feat_detached.unsqueeze(0).clone(),
            y_p=y_detached.clone() if y_detached is not None else None,
            raw_x=raw_detached.clone() if raw_detached is not None else None,
        )
        self.prototypes.append(new_proto)
        self._invalidate_cache()
        work_matrix = torch.cat([work_matrix, feat_detached.unsqueeze(0)], dim=0)
        return new_proto, work_matrix

    def register_task_batch(
        self,
        features: torch.Tensor,
        routing_dists: torch.Tensor,
        all_expert_outs: torch.Tensor,
        task_id: int,
        labels: Optional[torch.Tensor] = None,
        raw_inputs: Optional[torch.Tensor] = None,
        owner_expert: Optional[int] = None,
    ) -> None:
        """
        Registers representative samples from a batch into prototype memory.

        `owner_expert` records which expert was trained for this task so the
        prototype can anchor inference routing to it explicitly, independently of
        whatever the (possibly collapsed) router stored in `r_p`.
        """
        if features.size(0) == 0:
            return
        if self.distance_threshold is None:
            # Scale-free calibration (see _auto_threshold): a fixed absolute
            # threshold is meaningless across feature spaces — measured nearest
            # prototype distances on Split-CIFAR are ~6.5-7.2, so the old 0.5
            # effectively turned every sample into its own prototype.
            self.distance_threshold = self._auto_threshold(features)
            print(
                f"    [prototype] auto distance threshold = "
                f"{self.distance_threshold:.3f}"
            )
        work_matrix = (
            torch.stack([p.v_p for p in self.prototypes], dim=0)
            if self.prototypes
            else None
        )
        for i in range(features.size(0)):
            lbl = labels[i] if labels is not None else None
            raw = raw_inputs[i] if raw_inputs is not None else None
            _, work_matrix = self._update_or_create_with_matrix(
                features[i],
                routing_dists[i],
                all_expert_outs[i],
                task_id,
                lbl,
                raw,
                work_matrix,
                owner_expert=owner_expert,
            )

    @torch.no_grad()
    def _auto_threshold(self, features: torch.Tensor, quantile: float = 1.0) -> float:
        """
        Scale-free threshold calibration: the median nearest-neighbour distance
        inside the registration batch, times `quantile`. Points closer than the
        typical local spacing merge (via EMA chaining), so a dense region
        collapses to a few centroids while distinct clusters stay apart — unlike
        a fixed absolute threshold, which is meaningless across feature spaces.
        """
        x = features.detach().float()
        if x.size(0) < 2:
            return 0.5
        dists = torch.cdist(x, x)
        dists.fill_diagonal_(float("inf"))
        nearest = dists.min(dim=1).values
        value = float(nearest.median().item()) * quantile
        return max(value, 1e-6)

    @staticmethod
    def _label_of(proto: Prototype) -> int:
        if proto.y_p is not None and proto.y_p.numel() > 0:
            return int(proto.y_p[0].item())
        return -1

    @torch.no_grad()
    def refresh_anchors(self, model: Any) -> int:
        """
        Recomputes the routing (`r_p`) and expert-output (`o_p`) anchors of every
        prototype with the CURRENT model, e.g. after joint calibration.

        Rationale: the stability losses anchor the next task's training to these
        targets. If they were recorded before calibration (which unfreezes and
        updates all experts), the targets are stale and the stability MSE fights
        the calibration improvements. Refreshing them keeps the anchor semantics
        ("do not change what the calibrated model does on old prototypes")
        consistent. Returns the number of refreshed prototypes.
        """
        if self.is_empty():
            return 0
        device = next(model.parameters()).device
        v_mat = self.get_prototype_matrix(device)
        if v_mat is None:
            return 0
        routing = model.router.get_full_distribution(v_mat)
        outputs = model.get_all_expert_outputs(v_mat)  # [P, N, C]
        for i, proto in enumerate(self.prototypes):
            proto.r_p = routing[i].detach().cpu()
            proto.o_p = outputs[i].detach().cpu()
        self._invalidate_cache()
        return len(self.prototypes)

    def _prune_or_merge_least_used(self) -> None:
        """
        Evicts the least-used prototype from the most overrepresented group.
        With `max_prototypes_per_class` the group is (task, class), which keeps a
        per-class budget; otherwise it is the task (the original behaviour).
        """
        if not self.prototypes:
            return
        from collections import Counter

        if self.max_prototypes_per_class is not None:
            groups = Counter(
                (p.task_id, self._label_of(p)) for p in self.prototypes
            )
            overrepresented = max(groups, key=groups.get)
            candidates = [
                (i, p.count)
                for i, p in enumerate(self.prototypes)
                if (p.task_id, self._label_of(p)) == overrepresented
            ]
        else:
            task_counts = Counter([p.task_id for p in self.prototypes])
            overrepresented_task = max(task_counts, key=task_counts.get)
            candidates = [
                (i, p.count)
                for i, p in enumerate(self.prototypes)
                if p.task_id == overrepresented_task
            ]
        min_idx = min(candidates, key=lambda x: x[1])[0]
        del self.prototypes[min_idx]
        self._invalidate_cache()

    def sync_on_merge(self, idx1: int, idx2: int) -> None:
        """
        Synchronizes historical prototype distributions and output anchors when expert idx2
        is merged into idx1:
        1. r_p[idx1] <- r_p[idx1] + r_p[idx2]
        2. o_p[idx1] <- 0.5 * (o_p[idx1] + o_p[idx2])
        3. Removes idx2 column from all r_p and o_p tensors.
        """
        for proto in self.prototypes:
            n = proto.r_p.size(0)
            if idx1 < n and idx2 < n:
                proto.r_p[idx1] = proto.r_p[idx1] + proto.r_p[idx2]
                proto.o_p[idx1] = 0.5 * (proto.o_p[idx1] + proto.o_p[idx2])
                indices = [i for i in range(n) if i != idx2]
                proto.r_p = proto.r_p[indices]
                proto.o_p = proto.o_p[indices]
                sum_r = proto.r_p.sum()
                if sum_r > 0:
                    proto.r_p = proto.r_p / sum_r
            elif idx2 < n:
                indices = [i for i in range(n) if i != idx2]
                proto.r_p = proto.r_p[indices]
                proto.o_p = proto.o_p[indices]
                sum_r = proto.r_p.sum()
                if sum_r > 0:
                    proto.r_p = proto.r_p / sum_r
        self._invalidate_cache()

    def sync_on_prune(self, prune_idx: int) -> None:
        """
        Synchronizes prototype distributions when an unused expert at prune_idx is pruned.
        """
        for proto in self.prototypes:
            n = proto.r_p.size(0)
            if prune_idx < n:
                indices = [i for i in range(n) if i != prune_idx]
                proto.r_p = proto.r_p[indices]
                proto.o_p = proto.o_p[indices]
                sum_r = proto.r_p.sum()
                if sum_r > 0:
                    proto.r_p = proto.r_p / sum_r

    def compute_stability_losses(
        self,
        model: Any,
        lambda_r: float = 1.0,
        lambda_e: float = 1.0,
        lambda_enc: float = 0.0,
        return_enc: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes the stability losses:
        L_router_stab = KL( g_old(v_p) || g_new(v_p) )
        L_expert_stab = MSE( E_old(v_p), E_new(v_p) )
        L_enc_stab = MSE( encoder(raw_x), v_p ) (if return_enc=True and lambda_enc > 0)
        Vectorized across all prototypes and active experts.
        """
        device = next(model.parameters()).device
        if self.is_empty():
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            if return_enc:
                return zero, zero, zero
            return zero, zero

        curr_num_experts = model.num_experts

        # 1. Router stability loss on cached anchor matrices (same ordering as
        #    the historical targets).
        anchor = self.get_router_anchor_matrices(curr_num_experts, device)
        if anchor is None:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            if return_enc:
                return zero, zero, zero
            return zero, zero
        vp_mat, rp_targets = anchor
        g_new_dist = model.router.get_full_distribution(vp_mat)  # [M, N]

        kl_router = F.kl_div(
            torch.log(g_new_dist + 1e-9),
            rp_targets,
            reduction="batchmean",
            log_target=False,
        )
        l_router_stab = torch.clamp(kl_router, min=0.0) * lambda_r

        # 2. Expert stability loss, vectorized weight/target selection from the
        #    cached matrices (no per-prototype .item() host syncs).
        expert_losses = []
        v_mat = self.get_prototype_matrix(device)
        r_mat = self.get_routing_matrix(curr_num_experts, device)
        o_mat = self.get_output_matrix(curr_num_experts, device)
        for i in range(curr_num_experts):
            mask = r_mat[:, i] > 0.01
            if not bool(mask.any()):
                continue
            sub_vp = v_mat[mask]
            target_out = o_mat[mask, i]
            w_tensor = r_mat[mask, i].unsqueeze(-1)
            e_curr_out = model.experts[i](sub_vp, track_usage=False)  # [M, C]
            num_classes = target_out.size(-1)
            mse_unreduced = F.mse_loss(e_curr_out, target_out, reduction="none")
            weighted_loss = (mse_unreduced * w_tensor).sum() / (
                len(self.prototypes) * num_classes
            )
            expert_losses.append(weighted_loss)

        if expert_losses:
            l_expert_stab = torch.stack(expert_losses).sum() * lambda_e
        else:
            l_expert_stab = torch.tensor(0.0, device=device)

        if return_enc:
            l_enc_stab = self.compute_encoder_stability_loss(
                model.encoder, lambda_enc=lambda_enc
            )
            return l_router_stab, l_expert_stab, l_enc_stab

        return l_router_stab, l_expert_stab

    def compute_encoder_stability_loss(
        self,
        encoder: nn.Module,
        lambda_enc: float = 1.0,
    ) -> torch.Tensor:
        """
        Calculates L_encoder_stab: feature-space distillation anchoring the trainable encoder
        output for stored raw exemplar images to their historical prototype targets v_p.
        Prevents representation drift during continual training with an unfrozen encoder.
        """
        device = next(encoder.parameters()).device
        if self.is_empty() or lambda_enc <= 0:
            return torch.tensor(0.0, device=device)

        raw_anchor = self.get_raw_anchor(device)
        if raw_anchor is None:
            return torch.tensor(0.0, device=device)
        raw_rows, owner_idx = raw_anchor
        v_mat = self.get_prototype_matrix(device)

        curr_feats = encoder(raw_rows)  # [M, D]
        per_row = ((curr_feats - v_mat[owner_idx]) ** 2).mean(dim=1)  # [M]
        # Average per prototype first, matching the previous per-prototype MSE mean.
        num_protos = len(self.prototypes)
        counts = torch.bincount(owner_idx, minlength=num_protos).clamp(min=1)
        sums = torch.zeros(num_protos, device=device).index_add_(0, owner_idx, per_row)
        return (sums / counts).mean() * lambda_enc

    def get_raw_exemplar_batch(
        self, device: torch.device
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Gathers all stored raw exemplar images and class labels across all prototypes.
        Returns:
            (all_raws, all_labels) on device, or None if no raw exemplars exist.
        """
        raws = []
        labels = []
        for p in self.prototypes:
            if (
                p.raw_x is not None
                and p.y_p is not None
                and p.raw_x.size(0) > 0
                and p.y_p.size(0) > 0
            ):
                min_len = min(p.raw_x.size(0), p.y_p.size(0))
                raws.append(p.raw_x[:min_len])
                labels.append(p.y_p[:min_len])
        if not raws:
            return None
        return torch.cat(raws, dim=0).to(device), torch.cat(labels, dim=0).to(device)

    def refresh_representations(self, encoder: nn.Module) -> None:
        """
        If encoder representation changes significantly, refreshes prototype vectors
        and exemplar feature buffers using raw exemplars.
        """
        device = next(encoder.parameters()).device
        encoder.eval()
        with torch.no_grad():
            for proto in self.prototypes:
                if proto.raw_x is not None and proto.raw_x.size(0) > 0:
                    raw_dev = proto.raw_x.to(device)
                    new_feats = encoder(raw_dev)
                    proto.x_p = new_feats.cpu()
                    proto.v_p = new_feats.mean(dim=0).cpu()
                elif proto.x_p is not None and proto.x_p.size(0) > 0:
                    proto.v_p = proto.x_p.mean(dim=0)
        self._invalidate_cache()

    def estimate_memory_footprint(self) -> Dict[str, Any]:
        """
        Estimates total parameter/element counts stored across all prototypes.
        """
        total_floats = 0
        for p in self.prototypes:
            total_floats += p.v_p.numel()
            total_floats += p.r_p.numel()
            total_floats += p.o_p.numel()
            if p.x_p is not None:
                total_floats += p.x_p.numel()
            if p.y_p is not None:
                total_floats += p.y_p.numel()
            if p.raw_x is not None:
                total_floats += p.raw_x.numel()
        return {
            "num_prototypes": len(self.prototypes),
            "total_elements": total_floats,
            "size_kb": (total_floats * 4) / 1024,
        }
