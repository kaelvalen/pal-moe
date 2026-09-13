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
    v_p: torch.Tensor                   # [feature_dim]
    r_p: torch.Tensor                   # [num_experts_at_creation]
    o_p: torch.Tensor                   # [num_experts_at_creation, num_classes] (expert output anchors)
    task_id: int
    count: int = 1
    x_p: Optional[torch.Tensor] = None   # [num_exemplars, feature_dim]
    y_p: Optional[torch.Tensor] = None   # [num_exemplars] (labels for Acc_proto)
    raw_x: Optional[torch.Tensor] = None # [num_exemplars, input_dim] (raw inputs for refresh)


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
    ):
        self.feature_dim = feature_dim
        self.distance_threshold = distance_threshold
        self.ema_alpha = ema_alpha
        self.max_prototypes = max_prototypes
        self.exemplars_per_proto = exemplars_per_proto

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

    def get_exemplar_batch(self, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Gathers all stored exemplar feature vectors and labels across all prototypes.
        Returns:
            (all_feats, all_labels) on device, or None if no labeled exemplars exist.
        """
        feats = []
        labels = []
        for p in self.prototypes:
            if p.x_p is not None and p.y_p is not None and p.x_p.size(0) > 0 and p.y_p.size(0) > 0:
                feats.append(p.x_p)
                labels.append(p.y_p)
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
        raw_detached = raw_input.detach().cpu().unsqueeze(0) if raw_input is not None else None

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
            proto.v_p = self.ema_alpha * proto.v_p + (1.0 - self.ema_alpha) * feat_detached
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
        """Prunes prototype with lowest assignment count to stay within budget."""
        counts = [p.count for p in self.prototypes]
        least_idx = int(torch.argmin(torch.tensor(counts)).item())
        del self.prototypes[least_idx]

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the two stability losses:
        L_router_stab = KL( g_old(v_p) || g_new(v_p) )
        L_expert_stab = MSE( E_old(v_p), E_new(v_p) )
        """
        device = next(model.parameters()).device
        if self.is_empty():
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero

        router_stab_losses = []
        expert_stab_losses = []

        curr_num_experts = model.num_experts

        for proto in self.prototypes:
            vp = proto.v_p.to(device).unsqueeze(0)  # [1, D]
            rp_old = proto.r_p.to(device)           # [N_old]
            op_old = proto.o_p.to(device)           # [N_old, C]
            n_old = rp_old.size(0)

            # 1. Router stability loss
            g_new_dist = model.router.get_full_distribution(vp).squeeze(0)  # [N_curr]

            if curr_num_experts > n_old:
                pad_size = curr_num_experts - n_old
                eps = 1e-4 / curr_num_experts
                rp_padded = torch.zeros(curr_num_experts, device=device)
                rp_padded[:n_old] = rp_old * (1.0 - eps * pad_size)
                rp_padded[n_old:] = eps
                rp_target = rp_padded
            else:
                rp_target = rp_old[:curr_num_experts]
                rp_target = rp_target / (rp_target.sum() + 1e-9)

            kl_router = F.kl_div(
                torch.log(g_new_dist + 1e-9).unsqueeze(0),
                rp_target.unsqueeze(0),
                reduction="batchmean",
                log_target=False,
            )
            router_stab_losses.append(torch.clamp(kl_router, min=0.0))

            # 2. Expert stability loss
            exp_loss_proto = torch.tensor(0.0, device=device)
            for i in range(min(n_old, curr_num_experts)):
                weight_i = rp_old[i]
                if weight_i > 0.01:
                    e_curr_out = model.experts[i](vp, track_usage=False).squeeze(0)
                    mse_i = F.mse_loss(e_curr_out, op_old[i])
                    exp_loss_proto = exp_loss_proto + weight_i * mse_i

            expert_stab_losses.append(exp_loss_proto)

        l_router_stab = torch.stack(router_stab_losses).mean() * lambda_r
        l_expert_stab = torch.stack(expert_stab_losses).mean() * lambda_e

        return l_router_stab, l_expert_stab

    def refresh_representations(self, encoder: nn.Module) -> None:
        """
        If encoder representation changes significantly, refreshes prototype vectors
        using raw exemplars or feature buffers.
        """
        device = next(encoder.parameters()).device
        encoder.eval()
        with torch.no_grad():
            for proto in self.prototypes:
                if proto.raw_x is not None and proto.raw_x.size(0) > 0:
                    raw_dev = proto.raw_x.to(device)
                    new_feats = encoder(raw_dev)
                    proto.v_p = new_feats.mean(dim=0).cpu()
                elif proto.x_p is not None and proto.x_p.size(0) > 0:
                    proto.v_p = proto.x_p.mean(dim=0)
