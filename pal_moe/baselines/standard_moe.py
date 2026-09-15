"""
Standard Fixed-Architecture MoE baseline.
Has a fixed number of experts (e.g. 4) and router, trained sequentially without prototype stability
or dynamic expansion mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List

from ..models.encoder import SharedEncoder
from ..models.router import DynamicRouter
from ..models.expert import MLPExpert
from ..models.moe import DynamicMoE


class StandardMoE(nn.Module):
    """
    Standard MoE baseline:
    - Fixed 4 experts
    - Top-k sparse routing
    - Naive sequential fine-tuning on continual tasks (no prototype memory, no stability loss)
    """

    def __init__(
        self,
        input_dim: int = 784,
        feature_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 10,
        num_experts: int = 4,
        top_k: int = 1,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        encoder = SharedEncoder(input_dim=input_dim, output_dim=feature_dim)
        router = DynamicRouter(
            input_dim=feature_dim, num_experts=num_experts, top_k=top_k
        )
        experts = [
            MLPExpert(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                expert_id=i,
            )
            for i in range(num_experts)
        ]
        self.moe = DynamicMoE(
            encoder=encoder, router=router, experts=experts, use_ema_encoder=False
        ).to(device)
        self.optimizer = torch.optim.Adam(self.moe.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe(x)

    def train_task(
        self, task_id: int, train_loader: Any, epochs: int = 5
    ) -> Dict[str, Any]:
        self.moe.train()
        losses = []
        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.moe(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

        return {"task_id": task_id, "loss": sum(losses) / max(len(losses), 1)}
