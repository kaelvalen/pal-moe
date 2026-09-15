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
    x_p: Optional[torch.Tensor] = None  # [num_exemplars, feature_dim]
    y_p: Optional[torch.Tensor] = None  # [num_exemplars] (labels for Acc_proto)
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
        distance_threshold: float = 0.5,
        ema_alpha: float = 0.9,
        max_prototypes: int = 50,
        exemplars_per_proto: int = 5,
        store_raw: bool = False,
    ):
        self.feature_dim = feature_dim
        self.distance_threshold = distance_threshold
        self.ema_alpha = ema_alpha
        self.max_prototypes = max_prototypes
        self.exemplars_per_proto = exemplars_per_proto
        self.store_raw = store_raw

        self.prototypes: List[Prototype] = []

    def __len__(self) -> int:
        return len(self.prototypes)

    def is_empty(self) -> bool:
        return len(self.prototypes) == 0

    def get_prototype_matrix(self, device: torch.device) -> Optional[torch.Tensor]:
        """Returns [P, feature_dim] matrix of all stored prototype vectors."""
        if self.is_empty():
            return None
        return torch.stack([p.v_p.to(device) for p in self.prototypes], dim=0)

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
    ) -> Prototype:
        """
        Processes a single feature vector:
        - If close to an existing prototype (<= distance_threshold), updates via EMA.
        - Otherwise, instantiates a new prototype.
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
                count=1,
                x_p=feat_detached.unsqueeze(0).clone(),
                y_p=y_detached.clone() if y_detached is not None else None,
                raw_x=raw_detached.clone() if raw_detached is not None else None,
            )
            self.prototypes.append(new_proto)
            return new_proto

        # Find closest existing prototype
        p_mat = torch.stack([p.v_p for p in self.prototypes], dim=0)
        dists = torch.norm(p_mat - feat_detached.unsqueeze(0), p=2, dim=1)
        min_dist, closest_idx = torch.min(dists, dim=0)

        if min_dist.item() <= self.distance_threshold:
            # Update existing prototype with EMA
            proto = self.prototypes[closest_idx.item()]
            proto.v_p = (
                self.ema_alpha * proto.v_p + (1.0 - self.ema_alpha) * feat_detached
            )
            proto.count += 1
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
            return proto
        else:
            # Check capacity before creating new
            if len(self.prototypes) >= self.max_prototypes:
                self._prune_or_merge_least_used()

            new_proto = Prototype(
                v_p=feat_detached.clone(),
                r_p=routing_detached.clone(),
                o_p=expert_detached.clone(),
                task_id=task_id,
                count=1,
                x_p=feat_detached.unsqueeze(0).clone(),
                y_p=y_detached.clone() if y_detached is not None else None,
                raw_x=raw_detached.clone() if raw_detached is not None else None,
            )
            self.prototypes.append(new_proto)
            return new_proto

    def register_task_batch(
        self,
        features: torch.Tensor,
        routing_dists: torch.Tensor,
        all_expert_outs: torch.Tensor,
        task_id: int,
        labels: Optional[torch.Tensor] = None,
        raw_inputs: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Registers representative samples from a batch into prototype memory.
        """
        for i in range(features.size(0)):
            lbl = labels[i] if labels is not None else None
            raw = raw_inputs[i] if raw_inputs is not None else None
            self.update_or_create_prototype(
                feat=features[i],
                routing_dist=routing_dists[i],
                expert_outputs=all_expert_outs[i],
                task_id=task_id,
                label=lbl,
                raw_input=raw,
            )

    def _prune_or_merge_least_used(self) -> None:
        """
        Prunes prototype from the most overrepresented task with lowest assignment count
        to maintain balanced task representation within the memory budget.
        """
        if not self.prototypes:
            return
        from collections import Counter

        task_counts = Counter([p.task_id for p in self.prototypes])
        overrepresented_task = max(task_counts, key=task_counts.get)
        candidates = [
            (i, p.count)
            for i, p in enumerate(self.prototypes)
            if p.task_id == overrepresented_task
        ]
        min_idx = min(candidates, key=lambda x: x[1])[0]
        del self.prototypes[min_idx]

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

        # Collect all vp and xp features to anchor the router heavily
        vp_list = []
        rp_targets_list = []

        for p_idx, proto in enumerate(self.prototypes):
            # Base prototype center
            vp_list.append(proto.v_p)
            rp_targets_list.append(proto.r_p)

            # Exemplar features (x_p) around the prototype
            if proto.x_p is not None:
                for i in range(proto.x_p.size(0)):
                    vp_list.append(proto.x_p[i])
                    rp_targets_list.append(proto.r_p)

        vp_mat = torch.stack(vp_list, dim=0).to(device)  # [P_total, D]
        P_total = vp_mat.size(0)

        # 1. Vectorized router stability loss
        g_new_dist = model.router.get_full_distribution(
            vp_mat
        )  # [P_total, curr_num_experts]

        rp_targets = torch.zeros(P_total, curr_num_experts, device=device)
        for p_idx, rp_old in enumerate(rp_targets_list):
            rp_old = rp_old.to(device)
            n_old = rp_old.size(0)
            if curr_num_experts > n_old:
                pad_size = curr_num_experts - n_old
                eps = 1e-4 / curr_num_experts
                rp_targets[p_idx, :n_old] = rp_old * (1.0 - eps * pad_size)
                rp_targets[p_idx, n_old:] = eps
            else:
                rp_target = rp_old[:curr_num_experts]
                rp_targets[p_idx] = rp_target / (rp_target.sum() + 1e-9)

        kl_router = F.kl_div(
            torch.log(g_new_dist + 1e-9),
            rp_targets,
            reduction="batchmean",
            log_target=False,
        )
        l_router_stab = torch.clamp(kl_router, min=0.0) * lambda_r

        # 2. Vectorized expert stability loss
        expert_losses = []
        for i in range(curr_num_experts):
            weights = []
            targets = []
            vp_for_expert = []
            for p_idx, proto in enumerate(self.prototypes):
                if i < proto.r_p.size(0):
                    w = proto.r_p[i].item()
                    if w > 0.01:
                        weights.append(w)
                        targets.append(proto.o_p[i])
                        vp_for_expert.append(proto.v_p)

            if vp_for_expert:
                sub_vp = torch.stack(vp_for_expert, dim=0).to(device)  # [M, D]
                e_curr_out = model.experts[i](sub_vp, track_usage=False)  # [M, C]
                target_out = torch.stack(targets, dim=0).to(device)  # [M, C]
                w_tensor = torch.tensor(weights, device=device).unsqueeze(-1)  # [M, 1]
                num_classes = target_out.size(-1)
                mse_unreduced = F.mse_loss(
                    e_curr_out, target_out, reduction="none"
                )  # [M, C]
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

        losses = []
        for proto in self.prototypes:
            if proto.raw_x is not None and proto.raw_x.size(0) > 0:
                raw_dev = proto.raw_x.to(device)
                curr_feats = encoder(raw_dev)  # [M, D]
                target_anchor = (
                    proto.v_p.to(device).unsqueeze(0).expand_as(curr_feats)
                )  # [M, D]
                losses.append(F.mse_loss(curr_feats, target_anchor))

        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean() * lambda_enc

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
