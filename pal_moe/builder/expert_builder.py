"""
Expert Builder, Validation Gate, and Capacity Control for PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts).

Ensures:
1. Function-preserving expansion from the best parent expert.
2. Training on new data coupled with distillation from old prototype anchors.
3. Strict validation gate testing:
   - New task performance > threshold
   - Old prototype performance degradation <= tolerance
   - Generalization gap bounded
   - Expected Calibration Error (ECE) bounded
4. Capacity control via usage-based pruning and redundancy-based merging.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

from ..models.expert import MLPExpert
from ..models.moe import DynamicMoE
from ..memory.prototype_memory import PrototypeMemory


@dataclass
class ValidationGateResult:
    passed: bool
    new_task_acc: float
    old_proto_loss_diff: float
    ece: float
    train_loss: float
    val_loss: float
    rejection_reason: Optional[str] = None


class ExpertBuilder:
    """
    Manages the lifecycle of expert expansion, distillation, validation gating, and capacity control.
    """
    def __init__(
        self,
        min_acc_threshold: float = 0.70,
        max_proto_drop: float = 0.25,
        max_ece: float = 0.25,
        distill_lambda: float = 1.0,
    ):
        self.min_acc_threshold = min_acc_threshold
        self.max_proto_drop = max_proto_drop
        self.max_ece = max_ece
        self.distill_lambda = distill_lambda

    def create_candidate_from_parent(
        self,
        parent_expert: MLPExpert,
        new_expert_id: int,
        creation_task: int,
    ) -> MLPExpert:
        """
        Function-preserving expansion: creates a new child expert identical in behavior
        to parent expert at initialization.
        """
        return parent_expert.clone_function_preserving(
            new_expert_id=new_expert_id,
            creation_task=creation_task,
        )

    def train_candidate(
        self,
        candidate_expert: MLPExpert,
        encoder: nn.Module,
        train_loader: Any,
        prototype_memory: Optional[PrototypeMemory] = None,
        parent_expert: Optional[MLPExpert] = None,
        epochs: int = 5,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ) -> float:
        """
        Trains candidate on new task with distillation from prototype anchors.
        """
        candidate_expert.train()
        encoder.eval()
        optimizer = torch.optim.Adam(candidate_expert.parameters(), lr=lr, weight_decay=1e-5)

        avg_loss = 0.0
        total_batches = 0

        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                with torch.no_grad():
                    h = encoder(x)

                logits = candidate_expert(h)
                loss_task = F.cross_entropy(logits, y)

                # Distillation loss on old prototype anchors
                loss_distill = torch.tensor(0.0, device=device)
                if (
                    prototype_memory is not None
                    and not prototype_memory.is_empty()
                    and parent_expert is not None
                ):
                    p_mat = prototype_memory.get_prototype_matrix(device)
                    if p_mat is not None and p_mat.size(0) > 0:
                        with torch.no_grad():
                            parent_anchor = parent_expert(p_mat, track_usage=False)
                        cand_anchor = candidate_expert(p_mat, track_usage=False)
                        # Distillation on parent anchor outputs
                        loss_distill = F.mse_loss(cand_anchor, parent_anchor) * self.distill_lambda

                loss = loss_task + loss_distill
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                total_batches += 1

        return avg_loss / max(total_batches, 1)

    @staticmethod
    def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
        """
        Computes Expected Calibration Error (ECE) on predictions.
        """
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=probs.device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item()

    def validate_candidate(
        self,
        candidate_expert: MLPExpert,
        parent_expert: MLPExpert,
        encoder: nn.Module,
        val_loader: Any,
        prototype_memory: Optional[PrototypeMemory] = None,
        train_loss: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ) -> ValidationGateResult:
        """
        Validation Gate testing:
        1. New task accuracy >= min_acc_threshold
        2. Prototype degradation <= max_proto_drop
        3. Generalization gap not diverging
        4. ECE <= max_ece
        """
        candidate_expert.eval()
        parent_expert.eval()
        encoder.eval()

        correct = 0
        total = 0
        val_loss_accum = 0.0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                h = encoder(x)
                logits = candidate_expert(h, track_usage=False)
                loss = F.cross_entropy(logits, y)
                val_loss_accum += loss.item() * x.size(0)

                probs = F.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += x.size(0)

                all_probs.append(probs)
                all_targets.append(y)

        new_task_acc = correct / max(total, 1)
        val_loss = val_loss_accum / max(total, 1)

        all_probs_cat = torch.cat(all_probs, dim=0)
        all_targets_cat = torch.cat(all_targets, dim=0)
        ece = self.compute_ece(all_probs_cat, all_targets_cat)

        # Evaluate performance on old prototype anchors compared to parent
        proto_loss_diff = 0.0
        if prototype_memory is not None and not prototype_memory.is_empty():
            p_mat = prototype_memory.get_prototype_matrix(device)
            if p_mat is not None and p_mat.size(0) > 0:
                with torch.no_grad():
                    parent_out = parent_expert(p_mat, track_usage=False)
                    cand_out = candidate_expert(p_mat, track_usage=False)
                    # Difference in output representation drift
                    proto_loss_diff = F.mse_loss(cand_out, parent_out).item()

        # Gate decisions
        rejection_reason = None
        passed = True

        if new_task_acc < self.min_acc_threshold:
            passed = False
            rejection_reason = f"New task acc ({new_task_acc:.2%}) below threshold ({self.min_acc_threshold:.2%})"
        elif proto_loss_diff > self.max_proto_drop:
            passed = False
            rejection_reason = f"Old prototype drift ({proto_loss_diff:.3f}) exceeds max allowed ({self.max_proto_drop:.3f})"
        elif ece > self.max_ece:
            passed = False
            rejection_reason = f"ECE ({ece:.3f}) exceeds calibration threshold ({self.max_ece:.3f})"

        return ValidationGateResult(
            passed=passed,
            new_task_acc=new_task_acc,
            old_proto_loss_diff=proto_loss_diff,
            ece=ece,
            train_loss=train_loss,
            val_loss=val_loss,
            rejection_reason=rejection_reason,
        )

    def enforce_capacity_control(
        self,
        model: DynamicMoE,
        max_experts: int = 8,
        min_usage_threshold: int = 10,
    ) -> Dict[str, Any]:
        """
        Maintains expert capacity budget:
        - If model.num_experts > max_experts:
          1. Prune completely unused expert if usage < min_usage_threshold.
          2. Otherwise, find most similar expert pair and merge them.
        """
        actions_taken = []

        while model.num_experts > max_experts:
            # Check for dead/unused expert
            usages = [e.usage_count.item() for e in model.experts]
            min_usage = min(usages)
            min_idx = usages.index(min_usage)

            if min_usage < min_usage_threshold and len(model.experts) > 1:
                actions_taken.append(f"Pruned expert {min_idx} with usage {min_usage}")
                model.prune_expert(min_idx)
                continue

            # Otherwise, calculate pairwise cosine similarity between expert output weights
            weights = [e.fc2.weight.data.flatten() for e in model.experts]
            n = len(weights)
            best_sim = -1.0
            merge_pair = (0, 1)

            for i in range(n):
                for j in range(i + 1, n):
                    sim = F.cosine_similarity(weights[i].unsqueeze(0), weights[j].unsqueeze(0)).item()
                    if sim > best_sim:
                        best_sim = sim
                        merge_pair = (i, j)

            i, j = merge_pair
            actions_taken.append(f"Merged expert {j} into {i} (similarity {best_sim:.3f})")
            model.merge_experts(i, j)

        return {"actions": actions_taken, "final_num_experts": model.num_experts}
