"""
Dynamic Router with dynamic expert growth, top-k sparsity, and entropy computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _project_to_null_space(
    w: torch.Tensor, basis: Optional[torch.Tensor], scale: float = 1.0
) -> torch.Tensor:
    """
    Null-space anchoring (Gram-Schmidt): projects direction `w` onto the null space
    of the span of `basis` rows (e.g., historical prototype centroids), so a newly
    added expert's routing region is orthogonal to (cannot overlap with) the
    historical experts' regions at initialization.
    Returns a unit direction scaled by `scale`.
    """
    if basis is None or basis.size(0) == 0:
        return w
    dirs = F.normalize(basis, p=2, dim=1)  # [B, D]
    # Bases rows may be mutually correlated: orthonormalize the span via QR
    # (columns of Q span exactly the same subspace as the basis rows).
    q, _ = torch.linalg.qr(dirs.t())  # [D, B] -> Q keeps B orthonormal cols
    q = q[:, : basis.size(0)]  # [D, B]
    coeffs = w.view(-1) @ q  # [B]
    w_orth = w.view(-1) - q @ coeffs  # [D]
    if w_orth.norm(p=2) > 1e-8:
        return F.normalize(w_orth, p=2, dim=0) * scale
    # Degenerate: w is fully inside the historical span. Re-sample a random
    # direction from the null space instead.
    rand = torch.randn_like(w.view(-1))
    coeffs_r = rand @ q
    w_orth = rand - q @ coeffs_r
    if w_orth.norm(p=2) > 1e-8:
        return F.normalize(w_orth, p=2, dim=0) * scale
    return w


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
        self.locked_experts = 0

        # Register hooks to mask gradients for locked historical experts
        self.gate.weight.register_hook(self._weight_backward_hook)
        self.gate.bias.register_hook(self._bias_backward_hook)

    def _weight_backward_hook(self, grad):
        if self.locked_experts > 0 and grad is not None:
            grad = grad.clone()
            grad[: self.locked_experts] = 0.0
        return grad

    def _bias_backward_hook(self, grad):
        if self.locked_experts > 0 and grad is not None:
            grad = grad.clone()
            grad[: self.locked_experts] = 0.0
        return grad

    def lock_historical_routing(self, num_locked: int):
        """Locks the routing probabilities for the first `num_locked` experts by zeroing their gradients."""
        self.locked_experts = min(num_locked, self.num_experts)

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
        null_space_basis: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Expands the router from N to N+1 experts.
        If parent_id is given, the new weight row is initialized smoothly from parent.
        If prototype_feat is given, it can also bias towards that feature direction.
        If null_space_basis is given (e.g. historical prototype centroids), the new
        row is additionally projected onto its null space so its routing region is
        disjoint from historical experts (null-space anchoring).
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
            nn.init.kaiming_uniform_(new_gate.weight.data[old_num : old_num + 1])
            new_gate.bias.data[old_num] = 0.0

        # Null-space anchoring: keep the new row orthogonal to historical regions
        if null_space_basis is not None and null_space_basis.size(0) > 0:
            new_gate.weight.data[old_num] = _project_to_null_space(
                new_gate.weight.data[old_num], null_space_basis, scale=3.0
            )

        self.gate = new_gate
        self.num_experts = new_num
        self.gate.weight.register_hook(self._weight_backward_hook)
        self.gate.bias.register_hook(self._bias_backward_hook)
        return old_num

    def prune_expert(self, prune_idx: int) -> None:
        """Removes the expert at prune_idx and shrinks the routing layer."""
        assert 0 <= prune_idx < self.num_experts
        assert self.num_experts > 1, "Cannot prune the only expert."

        old_num = self.num_experts
        new_num = old_num - 1

        keep_indices = [i for i in range(old_num) if i != prune_idx]
        new_gate = nn.Linear(self.input_dim, new_num, bias=True).to(
            self.gate.weight.device
        )

        new_gate.weight.data = self.gate.weight.data[keep_indices]
        new_gate.bias.data = self.gate.bias.data[keep_indices]

        self.gate = new_gate
        self.num_experts = new_num
        self.top_k = min(self.top_k, new_num)

        # The rebuilt gate is a fresh module: re-attach the historical-routing
        # lock hooks (they live on the old parameter tensors and would otherwise
        # be silently lost, un-freezing history after any prune/merge) and keep
        # the lock count consistent with the removed row.
        if prune_idx < self.locked_experts:
            self.locked_experts -= 1
        self.locked_experts = min(self.locked_experts, new_num)
        self.gate.weight.register_hook(self._weight_backward_hook)
        self.gate.bias.register_hook(self._bias_backward_hook)

    def merge_experts(self, idx1: int, idx2: int) -> int:
        """
        Merges idx2 into idx1 by averaging their routing weights, then prunes idx2.
        """
        assert (
            0 <= idx1 < self.num_experts
            and 0 <= idx2 < self.num_experts
            and idx1 != idx2
        )
        with torch.no_grad():
            self.gate.weight.data[idx1] = 0.5 * (
                self.gate.weight.data[idx1] + self.gate.weight.data[idx2]
            )
            self.gate.bias.data[idx1] = 0.5 * (
                self.gate.bias.data[idx1] + self.gate.bias.data[idx2]
            )
        self.prune_expert(idx2)
        return idx1


class DistanceRouter(nn.Module):
    """
    Non-parametric distance-based router.
    Routes instances based on cosine similarity to expert centroids.
    Because there are no learnable parameters in the routing mechanism,
    this strictly prevents catastrophic routing drift and gradient imbalance.
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_experts: int = 4,
        top_k: int = 1,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature

        # Expert centroids: [num_experts, input_dim]
        # We use register_buffer so they are saved in state_dict but NOT optimized by grad
        self.register_buffer("centroids", torch.randn(num_experts, input_dim))
        self.centroids = F.normalize(self.centroids, p=2, dim=1)
        self.locked_experts = 0

    def lock_historical_routing(self, num_locked: int):
        self.locked_experts = min(num_locked, self.num_experts)

    def forward(
        self, h: torch.Tensor, top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = top_k if top_k is not None else self.top_k
        k = min(k, self.num_experts)

        h_norm = F.normalize(h, p=2, dim=1)
        c_norm = F.normalize(self.centroids, p=2, dim=1)

        # Cosine similarity logits
        sim = torch.matmul(h_norm, c_norm.T)
        logits = sim / max(self.temperature, 1e-5)

        dense_probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(dense_probs, k=k, dim=-1)

        if k > 1:
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        routing_weights = torch.zeros_like(dense_probs)
        routing_weights.scatter_(-1, topk_idx, topk_probs)

        return routing_weights, topk_idx, logits

    def get_full_distribution(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = F.normalize(h, p=2, dim=1)
        c_norm = F.normalize(self.centroids, p=2, dim=1)
        sim = torch.matmul(h_norm, c_norm.T)
        logits = sim / max(self.temperature, 1e-5)
        return F.softmax(logits, dim=-1)

    @staticmethod
    def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
        eps = 1e-9
        return -torch.sum(probs * torch.log(probs + eps), dim=-1)

    def add_expert(
        self,
        parent_id: Optional[int] = None,
        prototype_feat: Optional[torch.Tensor] = None,
        null_space_basis: Optional[torch.Tensor] = None,
    ) -> int:
        old_num = self.num_experts
        new_num = old_num + 1

        new_centroids = torch.zeros(
            new_num, self.input_dim, device=self.centroids.device
        )
        new_centroids[:old_num] = self.centroids

        if prototype_feat is not None:
            new_centroids[old_num] = F.normalize(prototype_feat.view(-1), p=2, dim=0)
        elif parent_id is not None and 0 <= parent_id < old_num:
            noise = torch.randn_like(self.centroids[parent_id]) * 0.1
            new_centroids[old_num] = F.normalize(
                self.centroids[parent_id] + noise, p=2, dim=0
            )
        else:
            new_centroids[old_num] = F.normalize(
                torch.randn(self.input_dim, device=self.centroids.device), p=2, dim=0
            )

        # Null-space anchoring for the distance router as well
        if null_space_basis is not None and null_space_basis.size(0) > 0:
            new_centroids[old_num] = _project_to_null_space(
                new_centroids[old_num], null_space_basis, scale=1.0
            )

        self.centroids = new_centroids
        self.num_experts = new_num
        return old_num

    def prune_expert(self, prune_idx: int) -> None:
        assert 0 <= prune_idx < self.num_experts
        assert self.num_experts > 1, "Cannot prune the only expert."

        old_num = self.num_experts
        new_num = old_num - 1

        keep_indices = [i for i in range(old_num) if i != prune_idx]
        self.centroids = self.centroids[keep_indices]
        self.num_experts = new_num
        self.top_k = min(self.top_k, new_num)

        # Keep the historical lock aligned with the shrunk centroid bank.
        if prune_idx < self.locked_experts:
            self.locked_experts -= 1
        self.locked_experts = min(self.locked_experts, new_num)

    def merge_experts(self, idx1: int, idx2: int) -> int:
        assert (
            0 <= idx1 < self.num_experts
            and 0 <= idx2 < self.num_experts
            and idx1 != idx2
        )

        c_merged = 0.5 * (self.centroids[idx1] + self.centroids[idx2])
        self.centroids[idx1] = F.normalize(c_merged, p=2, dim=0)
        self.prune_expert(idx2)
        return idx1

    def update_active_centroid(self, h: torch.Tensor, momentum: float = 0.99):
        """Updates the centroid of the ACTIVE (unlocked) expert using an EMA of incoming features."""
        # Only the newest expert gets updated in our strictly frozen architecture
        active_idx = self.num_experts - 1
        if active_idx >= self.locked_experts:
            with torch.no_grad():
                h_mean = h.mean(dim=0)
                h_mean = F.normalize(h_mean, p=2, dim=0)
                c_old = self.centroids[active_idx]
                c_new = momentum * c_old + (1.0 - momentum) * h_mean
                self.centroids[active_idx] = F.normalize(c_new, p=2, dim=0)


class AttentionRouter(nn.Module):
    """
    Cosine-attention router: every expert owns a learnable key vector and routing
    logits are scaled cosine similarities between the (optionally projected)
    representation and the keys:

        score_i = <normalize(q(h)), normalize(k_i)> / temperature

    Why this is not just "another linear layer": with a linear query projection
    the score is bilinear, so it is at least as expressive as the linear gate, but
    the L2 normalization bounds the logits (their scale is set by `temperature`
    instead of the feature norm) and the keys give each expert an explicit
    geometric anchor that expansion can initialize from prototypes. This is the
    attention-style variant of the router API: same forward contract, same
    dynamic add/prune/merge, same historical-routing lock.
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_experts: int = 4,
        top_k: int = 1,
        temperature: float = 0.1,
        query_dim: Optional[int] = None,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.noise_std = noise_std
        self.locked_experts = 0

        q_dim = query_dim or input_dim
        self.query_dim = q_dim
        self.query = (
            nn.Identity()
            if q_dim == input_dim
            else nn.Linear(input_dim, q_dim, bias=False)
        )
        self.keys = nn.Parameter(torch.randn(num_experts, q_dim) / q_dim**0.5)
        self.keys.register_hook(self._key_backward_hook)

    def _key_backward_hook(self, grad):
        if self.locked_experts > 0 and grad is not None:
            grad = grad.clone()
            grad[: self.locked_experts] = 0.0
        return grad

    def lock_historical_routing(self, num_locked: int):
        """Freezes the routing keys of the first `num_locked` experts."""
        self.locked_experts = min(num_locked, self.num_experts)

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.query(h), p=2, dim=-1)
        k = F.normalize(self.keys, p=2, dim=-1)
        logits = (q @ k.t()) / max(self.temperature, 1e-5)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        return logits

    def forward(
        self, h: torch.Tensor, top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = top_k if top_k is not None else self.top_k
        k = min(k, self.num_experts)

        logits = self._logits(h)
        dense_probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(dense_probs, k=k, dim=-1)
        if k > 1:
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        routing_weights = torch.zeros_like(dense_probs)
        routing_weights.scatter_(-1, topk_idx, topk_probs)
        return routing_weights, topk_idx, logits

    def get_full_distribution(self, h: torch.Tensor) -> torch.Tensor:
        return F.softmax(self._logits(h), dim=-1)

    @staticmethod
    def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
        eps = 1e-9
        return -torch.sum(probs * torch.log(probs + eps), dim=-1)

    def add_expert(
        self,
        parent_id: Optional[int] = None,
        prototype_feat: Optional[torch.Tensor] = None,
        null_space_basis: Optional[torch.Tensor] = None,
    ) -> int:
        """Expands the key bank from N to N+1 experts (same contract as DynamicRouter)."""
        old_num = self.num_experts
        old_keys = self.keys.data
        q_dim = old_keys.size(1)
        new_keys = torch.randn(old_num + 1, q_dim, device=old_keys.device) * q_dim**-0.5
        new_keys[:old_num] = old_keys

        if prototype_feat is not None:
            feat = prototype_feat.view(1, -1)
            direction = F.normalize(self.query(feat).view(-1), p=2, dim=0)
            new_keys[old_num] = direction
        elif parent_id is not None and 0 <= parent_id < old_num:
            noise = torch.randn_like(old_keys[parent_id]) * 0.01
            new_keys[old_num] = old_keys[parent_id] + noise
        else:
            nn.init.normal_(new_keys[old_num], std=q_dim**-0.5)

        if null_space_basis is not None and null_space_basis.size(0) > 0:
            projected = _project_to_null_space(
                new_keys[old_num], null_space_basis, scale=1.0
            )
            if projected.norm(p=2) > 1e-8:
                new_keys[old_num] = F.normalize(projected, p=2, dim=0)

        self.keys = nn.Parameter(new_keys)
        self.keys.register_hook(self._key_backward_hook)
        self.num_experts = old_num + 1
        return old_num

    def prune_expert(self, prune_idx: int) -> None:
        assert 0 <= prune_idx < self.num_experts
        assert self.num_experts > 1, "Cannot prune the only expert."
        keep = [i for i in range(self.num_experts) if i != prune_idx]
        self.keys = nn.Parameter(self.keys.data[keep].clone())
        self.keys.register_hook(self._key_backward_hook)
        self.num_experts -= 1
        self.top_k = min(self.top_k, self.num_experts)

        # Keep the historical lock aligned with the shrunk key bank.
        if prune_idx < self.locked_experts:
            self.locked_experts -= 1
        self.locked_experts = min(self.locked_experts, self.num_experts)

    def merge_experts(self, idx1: int, idx2: int) -> int:
        assert (
            0 <= idx1 < self.num_experts
            and 0 <= idx2 < self.num_experts
            and idx1 != idx2
        )
        with torch.no_grad():
            merged = 0.5 * (self.keys.data[idx1] + self.keys.data[idx2])
            self.keys.data[idx1] = F.normalize(merged, p=2, dim=0)
        self.prune_expert(idx2)
        return idx1
