"""
Quantitative Expert Creation Trigger for PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts).

Evaluates whether existing experts are sufficient using the composite metric:
    S(x) = alpha * L_task(best_expert) + beta * H(g(x)) + gamma * d(x, P) - delta * max_i(confidence_i)

If E[S(x)] > tau, a new expert candidate is warranted.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F


@dataclass
class TriggerEvaluationResult:
    should_trigger: bool
    composite_score: float
    loss_best_expert: float
    router_entropy: float
    proto_distance: float
    max_confidence: float
    best_parent_expert_idx: int


class QuantitativeTrigger:
    """
    Computes S(x) over a batch or window to decide when to expand the expert pool.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        delta: float = 0.5,
        threshold_tau: float = 1.2,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.threshold_tau = threshold_tau

    def evaluate(
        self,
        model: Any,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        prototype_memory: Optional[Any] = None,
    ) -> TriggerEvaluationResult:
        """
        Calculates S(x) across batch:
            - L_task(best expert): CE loss or prediction entropy if y is None
            - H(g(x)): Router entropy
            - d(x, P): Minimum distance to prototype memory
            - max_confidence: Maximum confidence among existing experts
        """
        # Inference-only: restore the caller's training mode afterwards, so a
        # trigger check can never silently leave the model in eval mode for the
        # rest of the task's training loop.
        was_training = model.training
        model.eval()
        try:
            return self._evaluate_impl(model, x, y, prototype_memory)
        finally:
            model.train(was_training)

    def _evaluate_impl(
        self,
        model: Any,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        prototype_memory: Optional[Any],
    ) -> TriggerEvaluationResult:
        with torch.no_grad():
            h = model.get_routing_features(x)  # [B, D]
            routing_weights, _, _ = model.router(h)  # [B, N]
            router_entropy = model.router.compute_entropy(routing_weights).mean().item()

            num_experts = model.num_experts
            expert_losses = []
            expert_max_confs = []

            for i in range(num_experts):
                exp_logits = model.experts[i](h, track_usage=False)  # [B, C]
                exp_probs = F.softmax(exp_logits, dim=-1)
                max_conf_i = exp_probs.max(dim=-1).values.mean().item()
                expert_max_confs.append(max_conf_i)

                if y is not None:
                    loss_i = F.cross_entropy(exp_logits, y).item()
                else:
                    # Unsupervised: entropy of expert's prediction (lower entropy = higher confidence)
                    loss_i = (
                        -(exp_probs * torch.log(exp_probs + 1e-9))
                        .sum(dim=-1)
                        .mean()
                        .item()
                    )
                expert_losses.append(loss_i)

            # Best expert has minimum loss
            best_parent_idx = int(torch.tensor(expert_losses).argmin().item())
            loss_best_expert = expert_losses[best_parent_idx]
            max_confidence = max(expert_max_confs)

            # Distance to prototype memory
            if prototype_memory is not None and not prototype_memory.is_empty():
                proto_dists = prototype_memory.compute_min_distance(h)
                mean_proto_dist = proto_dists.mean().item()
            else:
                mean_proto_dist = 1.0  # High distance if no prototypes exist yet

            # S(x) = alpha * L_task + beta * H(g) + gamma * d(x, P) - delta * max_conf
            composite_score = (
                self.alpha * loss_best_expert
                + self.beta * router_entropy
                + self.gamma * mean_proto_dist
                - self.delta * max_confidence
            )

            should_trigger = bool(composite_score > self.threshold_tau)

            return TriggerEvaluationResult(
                should_trigger=should_trigger,
                composite_score=composite_score,
                loss_best_expert=loss_best_expert,
                router_entropy=router_entropy,
                proto_distance=mean_proto_dist,
                max_confidence=max_confidence,
                best_parent_expert_idx=best_parent_idx,
            )


class AlwaysTrigger:
    """
    Expansion trigger that always fires on a new task.

    Used with ``--expand_every_task`` in the supervised task-incremental
    protocol: every task gets its own expert (up to ``--max_experts``), which
    removes expert sharing on long horizons - 20-task CIFAR-100 with a
    six-expert cap forced the later tasks onto already-used experts and cost
    accuracy. The parent for the function-preserving expansion is the newest
    expert (the one that just finished the previous task).
    """

    def evaluate(
        self,
        model: Any,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        prototype_memory: Optional[Any] = None,
    ) -> TriggerEvaluationResult:
        return TriggerEvaluationResult(
            should_trigger=True,
            composite_score=1.0,
            loss_best_expert=0.0,
            router_entropy=0.0,
            proto_distance=0.0,
            max_confidence=0.0,
            best_parent_expert_idx=max(0, model.num_experts - 1),
        )
