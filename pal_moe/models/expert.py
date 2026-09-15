"""
Expert architectures for PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts).

Supports:
- MLP Expert with residual adapter capacity
- Function-preserving expansion (exact initialization matching parent output: E_child(h) == E_parent(h))
- Tracking usage frequency and lineage for capacity control
"""

import copy
import torch
import torch.nn as nn
from typing import Optional


class MLPExpert(nn.Module):
    """
    Expert network E_i that maps representation h(x) -> task logits R^C.
    Equipped with a primary pathway and an expandable residual adapter.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 10,
        expert_id: int = 0,
        creation_task: int = 0,
        parent_id: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.expert_id = expert_id
        self.creation_task = creation_task
        self.parent_id = parent_id

        # Usage statistics for capacity control
        self.register_buffer("usage_count", torch.zeros(1, dtype=torch.long))

        # Base pathway
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Residual adapter branch (can be expanded/activated for new capacity)
        self.adapter_down = nn.Linear(input_dim, hidden_dim // 2, bias=False)
        self.adapter_act = nn.GELU()
        self.adapter_up = nn.Linear(hidden_dim // 2, num_classes, bias=False)
        # Initialize adapter_up to ZERO for function-preserving expansion
        nn.init.zeros_(self.adapter_up.weight)

    def forward(self, h: torch.Tensor, track_usage: bool = True) -> torch.Tensor:
        if self.training and track_usage:
            self.usage_count += int(h.size(0))

        base_out = self.fc2(self.dropout(self.act1(self.fc1(h))))
        adapter_out = self.adapter_up(self.adapter_act(self.adapter_down(h)))
        return base_out + adapter_out

    def clone_function_preserving(
        self, new_expert_id: int, creation_task: int, freeze_base: bool = False
    ) -> "MLPExpert":
        """
        Creates a new expert E_child via Function-Preserving Expansion from this expert (parent):
        1. Clones fc1 and fc2 weights directly from parent.
        2. Adapter branch is zero-initialized.
        3. If freeze_base is True, freezes fc1/fc2 so only the adapter trains (Parameter-Efficient).
        Therefore: E_child(h) == E_parent(h) identically for all h at initialization.
        """
        child = MLPExpert(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            expert_id=new_expert_id,
            creation_task=creation_task,
            parent_id=self.expert_id,
        ).to(self.fc1.weight.device)

        # Copy primary pathway weights
        child.fc1.load_state_dict(self.fc1.state_dict())
        child.fc2.load_state_dict(self.fc2.state_dict())

        # Parameter efficiency: Freeze base pathway to prevent parameter explosion
        if freeze_base:
            for p in child.fc1.parameters():
                p.requires_grad = False
            for p in child.fc2.parameters():
                p.requires_grad = False

        # Copy adapter_down representation if desired, ensure adapter_up is strictly zero
        child.adapter_down.load_state_dict(self.adapter_down.state_dict())
        nn.init.zeros_(child.adapter_up.weight)

        return child


class ExpertAdapter(MLPExpert):
    """Alias for MLPExpert with adapter."""

    pass
