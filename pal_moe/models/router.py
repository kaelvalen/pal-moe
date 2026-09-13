"""
Dynamic Router with dynamic expert growth, top-k sparsity, and entropy computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DynamicRouter(nn.Module):
    """
    Dynamic Router g(x) mapping representation h in R^d to expert selection weights.
    Supports:
    - Top-k selection (top-1, top-2, etc.)
    - Router entropy calculation H(g(x))
    - Dynamic expansion: adding new expert routing weights
    - Merging and pruning expert weights
    - Routing stability evaluation
    """
    def __init__(
        self,
        input_dim: int = 128,
        num_experts: int = 4,
        top_k: int = 1,
        temperature: float = 1.0,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.noise_std = noise_std

        # Linear projection from representation to expert logits
        self.gate = nn.Linear(input_dim, num_experts, bias=True)

    def forward(
        self, h: torch.Tensor, top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [batch_size, input_dim]
            top_k: override top_k if specified

        Returns:
            routing_weights: [batch_size, num_experts] (sparse or softmax)
            topk_indices: [batch_size, k]
            raw_logits: [batch_size, num_experts]
        """
        k = top_k if top_k is not None else self.top_k
        k = min(k, self.num_experts)

        logits = self.gate(h) / max(self.temperature, 1e-5)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Full softmax distribution preserves gradients to all router parameters for top-1 routing
        dense_probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(dense_probs, k=k, dim=-1)

        # Normalize top-k probabilities to sum to 1 when k > 1
        if k > 1:
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Create sparse routing weights tensor
        routing_weights = torch.zeros_like(dense_probs)
        routing_weights.scatter_(-1, topk_idx, topk_probs)

        return routing_weights, topk_idx, logits

    def get_full_distribution(self, h: torch.Tensor) -> torch.Tensor:
        """Returns dense softmax distribution over all experts."""
        logits = self.gate(h) / max(self.temperature, 1e-5)
        return F.softmax(logits, dim=-1)

    @staticmethod
    def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
        """
        Calculates router entropy H(g(x)) = - sum_i g_i log(g_i + eps).
        Args:
            probs: [batch_size, num_experts]
        Returns:
            entropy: [batch_size]
        """
        eps = 1e-9
        return -torch.sum(probs * torch.log(probs + eps), dim=-1)

    def add_expert(
        self,
        parent_id: Optional[int] = None,
        prototype_feat: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Expands the router from N to N+1 experts.
        If parent_id is given, the new weight row is initialized smoothly from parent.
        If prototype_feat is given, it can also bias towards that feature direction.
        Returns:
            new_expert_id (int)
        """
        old_num = self.num_experts
        new_num = old_num + 1

        old_weight = self.gate.weight.data
        old_bias = self.gate.bias.data

        new_gate = nn.Linear(self.input_dim, new_num, bias=True).to(old_weight.device)

        # Copy existing weights
        new_gate.weight.data[:old_num] = old_weight
        new_gate.bias.data[:old_num] = old_bias

        # Initialize new row
        if prototype_feat is not None:
            norm_feat = F.normalize(prototype_feat.view(-1), dim=0) * 3.0
            new_gate.weight.data[old_num] = norm_feat
            new_gate.bias.data[old_num] = 0.0
        elif parent_id is not None and 0 <= parent_id < old_num:
            # Inherit from parent with slight perturbation
            noise = torch.randn_like(old_weight[parent_id]) * 0.01
            new_gate.weight.data[old_num] = old_weight[parent_id] + noise
            new_gate.bias.data[old_num] = old_bias[parent_id]
        else:
            # Kaiming normal
            nn.init.kaiming_uniform_(new_gate.weight.data[old_num:old_num+1])
            new_gate.bias.data[old_num] = 0.0

        self.gate = new_gate
        self.num_experts = new_num
        return old_num

    def prune_expert(self, prune_idx: int) -> None:
        """Removes the expert at prune_idx and shrinks the routing layer."""
        assert 0 <= prune_idx < self.num_experts
        assert self.num_experts > 1, "Cannot prune the only expert."

        old_num = self.num_experts
        new_num = old_num - 1

        keep_indices = [i for i in range(old_num) if i != prune_idx]
        new_gate = nn.Linear(self.input_dim, new_num, bias=True).to(self.gate.weight.device)

        new_gate.weight.data = self.gate.weight.data[keep_indices]
        new_gate.bias.data = self.gate.bias.data[keep_indices]

        self.gate = new_gate
        self.num_experts = new_num
        self.top_k = min(self.top_k, new_num)

    def merge_experts(self, idx1: int, idx2: int) -> int:
        """
        Merges idx2 into idx1 by averaging their routing weights, then prunes idx2.
        """
        assert 0 <= idx1 < self.num_experts and 0 <= idx2 < self.num_experts and idx1 != idx2
        with torch.no_grad():
            self.gate.weight.data[idx1] = 0.5 * (self.gate.weight.data[idx1] + self.gate.weight.data[idx2])
            self.gate.bias.data[idx1] = 0.5 * (self.gate.bias.data[idx1] + self.gate.bias.data[idx2])
        self.prune_expert(idx2)
        return idx1
