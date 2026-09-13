"""
Evaluation metrics for Continual Learning and PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts).

Implements:
- Average Accuracy (ACC)
- Catastrophic Forgetting (F)
- Backward Transfer (BWT)
- Forward Transfer (FWT)
- Router Stability: KL divergence change on old prototypes
- Expert Specialization: Task-Expert Mutual Information I(Task; Expert)
- Expert Utilization: Normalized activation entropy
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from ..memory.prototype_memory import PrototypeMemory
from ..models.moe import DynamicMoE


@dataclass
class BenchmarkResult:
    method_name: str
    acc_matrix: np.ndarray             # [T, T]
    average_accuracy: float            # ACC_T
    forgetting: float                  # F
    backward_transfer: float           # BWT
    router_stability_kl: float         # mean KL divergence
    expert_specialization_mi: float    # I(Task; Expert)
    expert_utilization_entropy: float  # Normalized H(Expert)
    num_final_experts: int
    task_accuracies: List[float]


class ContinualEvaluator:
    """
    Tracks and computes continual learning benchmark metrics across task sequence.
    """
    def __init__(self, num_tasks: int, device: torch.device = torch.device("cpu")):
        self.num_tasks = num_tasks
        self.device = device
        # R[t, i] = accuracy on task i after finishing training on task t
        self.R = np.zeros((num_tasks, num_tasks), dtype=np.float32)

    def evaluate_task(self, model: nn.Module, test_loader: Any) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += x.size(0)
        return correct / max(total, 1)

    def evaluate_all_seen_tasks(
        self, model: nn.Module, current_task_id: int, tasks: List[Any]
    ) -> List[float]:
        """Evaluates model on all tasks up to current_task_id and updates matrix R."""
        task_accs = []
        for i in range(current_task_id + 1):
            acc = self.evaluate_task(model, tasks[i].test_loader)
            self.R[current_task_id, i] = acc
            task_accs.append(acc)
        return task_accs

    def compute_average_accuracy(self, after_task: Optional[int] = None) -> float:
        """A_T = (1/T) sum_{i=1}^T R_{T, i}"""
        t = (self.num_tasks - 1) if after_task is None else after_task
        return float(np.mean(self.R[t, : t + 1]))

    def compute_forgetting(self) -> float:
        """
        F = (1 / (T - 1)) * sum_{i=1}^{T-1} max_{t in {1,...,T-1}} (R_{t, i} - R_{T, i})
        """
        if self.num_tasks <= 1:
            return 0.0
        T = self.num_tasks - 1
        forgettings = []
        for i in range(T):
            max_past_acc = np.max(self.R[i:T, i])
            final_acc = self.R[T, i]
            forgettings.append(max(0.0, float(max_past_acc - final_acc)))
        return float(np.mean(forgettings))

    def compute_backward_transfer(self) -> float:
        """
        BWT = (1 / (T - 1)) * sum_{i=1}^{T-1} (R_{T, i} - R_{i, i})
        """
        if self.num_tasks <= 1:
            return 0.0
        T = self.num_tasks - 1
        bwts = [float(self.R[T, i] - self.R[i, i]) for i in range(T)]
        return float(np.mean(bwts))

    @staticmethod
    def compute_router_stability(
        model: DynamicMoE, prototype_memory: PrototypeMemory, device: torch.device
    ) -> float:
        """
        Computes mean KL divergence on prototype memory between original routing distributions
        and current routing distributions.
        """
        if prototype_memory.is_empty():
            return 0.0

        model.eval()
        kl_divs = []
        curr_num_experts = model.num_experts

        with torch.no_grad():
            for proto in prototype_memory.prototypes:
                vp = proto.v_p.to(device).unsqueeze(0)
                rp_old = proto.r_p.to(device)
                n_old = rp_old.size(0)

                g_curr = model.router.get_full_distribution(vp).squeeze(0)

                # Pad old distribution if needed
                if curr_num_experts > n_old:
                    pad_size = curr_num_experts - n_old
                    eps = 1e-5 / curr_num_experts
                    rp_target = torch.zeros(curr_num_experts, device=device)
                    rp_target[:n_old] = rp_old * (1.0 - eps * pad_size)
                    rp_target[n_old:] = eps
                else:
                    rp_target = rp_old[:curr_num_experts]
                    rp_target = rp_target / (rp_target.sum() + 1e-9)

                kl = F.kl_div(
                    torch.log(g_curr + 1e-9).unsqueeze(0),
                    rp_target.unsqueeze(0),
                    reduction="batchmean",
                    log_target=False,
                ).item()
                kl_divs.append(max(0.0, kl))

        return float(np.mean(kl_divs)) if kl_divs else 0.0

    @staticmethod
    def compute_expert_specialization_and_utilization(
        model: DynamicMoE, tasks: List[Any], device: torch.device
    ) -> Tuple[float, float]:
        """
        Calculates:
        1. Task-Expert Mutual Information I(Task; Expert):
           Measures how strongly experts specialize to specific tasks.
        2. Normalized Expert Utilization Entropy:
           H(Expert) / log2(N_experts), measuring balance and non-collapse.
        """
        model.eval()
        num_tasks = len(tasks)
        num_experts = model.num_experts

        # Count matrix C[task, expert]
        counts = np.zeros((num_tasks, num_experts), dtype=np.float64)

        with torch.no_grad():
            for t_idx, task in enumerate(tasks):
                for x, _ in task.test_loader:
                    x = x.to(device)
                    h = model.get_routing_features(x)
                    _, topk_idx, _ = model.router(h)
                    for k_idx in topk_idx.flatten().cpu().numpy():
                        if k_idx < num_experts:
                            counts[t_idx, k_idx] += 1

        total_counts = np.sum(counts)
        if total_counts == 0:
            return 0.0, 0.0

        # Joint distribution P(T, E)
        P_te = counts / total_counts
        P_t = np.sum(P_te, axis=1, keepdims=True)  # P(T)
        P_e = np.sum(P_te, axis=0, keepdims=True)  # P(E)

        # Mutual Information: I(T; E) = sum P(t, e) log2( P(t,e) / (P(t)P(e)) )
        eps = 1e-12
        P_indep = P_t @ P_e
        ratio = (P_te + eps) / (P_indep + eps)
        mi = np.sum(P_te * np.log2(np.maximum(ratio, eps)))
        mi = float(max(0.0, mi))

        # Normalized Entropy of Expert distribution H(E)
        P_e_flat = P_e.flatten()
        entropy_e = -np.sum(P_e_flat * np.log2(P_e_flat + eps))
        max_entropy = math.log2(max(num_experts, 2))
        normalized_utilization = float(min(1.0, max(0.0, entropy_e / max_entropy)))

        return mi, normalized_utilization
