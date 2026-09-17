"""
Elastic Weight Consolidation (EWC) baseline.
Computes diagonal Fisher Information matrix on past tasks to penalize changes to critical weights.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class EWC:
    """
    Online / Multi-task Elastic Weight Consolidation.
    L_ewc = L_new + sum_p (lambda_ewc / 2) * F_p * (theta_p - theta_p*)^2
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 500.0,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.fisher_matrices: list[dict[str, torch.Tensor]] = []
        self.star_params: list[dict[str, torch.Tensor]] = []

    def compute_fisher(self, data_loader: Any, num_samples: int = 200) -> None:
        """Computes diagonal empirical Fisher Information matrix for current task."""
        self.model.eval()
        fisher = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        params_star = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        samples_processed = 0
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += (p.grad.data**2) * x.size(0)

            samples_processed += x.size(0)
            if samples_processed >= num_samples:
                break

        for n in fisher:
            fisher[n] /= max(samples_processed, 1)

        self.fisher_matrices.append(fisher)
        self.star_params.append(params_star)

    def ewc_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for fisher, star in zip(self.fisher_matrices, self.star_params):
            for n, p in self.model.named_parameters():
                if n in fisher:
                    loss += (fisher[n] * (p - star[n]) ** 2).sum()
        return loss * (self.ewc_lambda / 2.0)

    def train_task(
        self, task_id: int, train_loader: Any, epochs: int = 5
    ) -> dict[str, Any]:
        self.model.train()
        losses = []
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                logits = self.model(x)
                loss_ce = F.cross_entropy(logits, y)
                loss_ewc = self.ewc_loss()
                loss = loss_ce + loss_ewc

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

        # Update Fisher for next tasks
        self.compute_fisher(train_loader)
        return {"task_id": task_id, "loss": sum(losses) / max(len(losses), 1)}
