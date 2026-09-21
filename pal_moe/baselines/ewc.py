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
    Multi-task / Online Elastic Weight Consolidation.
    L_ewc = L_new + sum_p (lambda_ewc / 2) * F_p * (theta_p - theta_p*)^2

    By default every task's Fisher matrix is kept and summed (the original
    behaviour, O(T * params) memory). With `online=True` a single running
    Fisher with gamma-decayed old mass is maintained instead (Schwarz et al.,
    2018), which is the standard constant-memory approximation.
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 500.0,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        online: bool = False,
        online_gamma: float = 0.9,
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.online = online
        self.online_gamma = online_gamma
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

        if self.online and self.fisher_matrices:
            # F_t <- gamma * F_{t-1} + F_t, with a single star-parameter set.
            previous = self.fisher_matrices[0]
            for n in fisher:
                fisher[n] = self.online_gamma * previous[n] + fisher[n]
            self.fisher_matrices[0] = fisher
            self.star_params[0] = params_star
        else:
            self.fisher_matrices.append(fisher)
            self.star_params.append(params_star)

    def ewc_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for fisher, star in zip(self.fisher_matrices, self.star_params):
            for n, p in self.model.named_parameters():
                if n in fisher:
                    loss += (fisher[n] * (p - star[n]) ** 2).sum()
        return loss * (self.ewc_lambda / 2.0)

    def memory_bytes(self) -> int:
        """Stored regularization state: one Fisher + one parameter snapshot per
        task (or a single pair with ``online=True``)."""
        total = 0
        for state in self.fisher_matrices + self.star_params:
            for tensor in state.values():
                total += tensor.numel() * tensor.element_size()
        return int(total)

    def train_task(
        self, task_id: int, train_loader: Any, epochs: int = 5
    ) -> dict[str, Any]:
        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        n_updates = 0
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True
                )
                self.optimizer.zero_grad()

                logits = self.model(x)
                loss_ce = F.cross_entropy(logits, y)
                loss_ewc = self.ewc_loss()
                loss = loss_ce + loss_ewc

                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach()
                n_updates += 1

        # Update Fisher for next tasks
        self.compute_fisher(train_loader)
        return {
            "task_id": task_id,
            "loss": float(total_loss.item()) / max(n_updates, 1),
        }
