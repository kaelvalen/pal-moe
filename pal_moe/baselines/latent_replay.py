"""
Latent replay baseline: a single-head classifier trained on stored latent
vectors instead of raw inputs.

This is the single-network counterpart of PAL-MoE's latent store and the
"Latent Replay" row of the component ablation: the frozen encoder produces
``h = f(x)`` once, the buffer keeps ``(h, y)`` pairs (no pixels), and the head
is trained with cross-entropy on the current batch plus a replay batch of
stored latents.

Interface matches :class:`pal_moe.baselines.replay.ReplayTrainer`. The trainer
takes the encoder and the head separately so it works both on raw inputs and on
the identity encoder of a feature-cached run:

    logits      = head(encoder(x))     # current batch
    replay_logits = head(h_stored)     # no encoder pass for stored latents

With a trainable encoder the stored latents drift as the encoder changes (the
stale-latent failure mode); freeze the encoder for the published comparison.
"""

from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffer import SampleBuffer


class LatentReplayTrainer:
    def __init__(
        self,
        model: nn.Module,
        encoder_fn: Callable[[torch.Tensor], torch.Tensor],
        head: nn.Module,
        buffer_size: int = 200,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        sampling: str = "recency",
    ):
        self.model = model
        self.encoder_fn = encoder_fn
        self.head = head
        self.buffer_size = buffer_size
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.buffer = SampleBuffer(buffer_size, mode=sampling)

    def update_buffer(self, train_loader: Any) -> None:
        """Stores `buffer_size // 5` latent vectors from the current task."""
        latents, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for x, y in train_loader:
                h = self.encoder_fn(x.to(self.device, non_blocking=True))
                latents.append(h.detach().cpu())
                labels.append(y)
        if not latents:
            return
        self.buffer.add_task(
            torch.cat(latents, dim=0),
            torch.cat(labels, dim=0),
            per_task_budget=max(1, self.buffer_size // 5),
        )

    def get_replay_batch(
        self, batch_size: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bx, by, _ = self.buffer.sample_tensors(batch_size, self.device)
        return bx, by

    def train_task(
        self, task_id: int, train_loader: Any, epochs: int = 5
    ) -> dict[str, Any]:
        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        n_updates = 0
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = (
                    x.to(self.device, non_blocking=True),
                    y.to(self.device, non_blocking=True),
                )
                self.optimizer.zero_grad()
                logits = self.head(self.encoder_fn(x))
                loss = F.cross_entropy(logits, y)

                rx, ry = self.get_replay_batch(batch_size=x.size(0) // 2)
                if rx is not None and ry is not None:
                    r_logits = self.head(rx)
                    loss = 0.5 * loss + 0.5 * F.cross_entropy(r_logits, ry)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach()
                n_updates += 1

        self.update_buffer(train_loader)
        return {
            "task_id": task_id,
            "loss": float(total_loss.item()) / max(n_updates, 1),
        }

    def memory_bytes(self) -> int:
        """Stored latent bytes (no raw inputs)."""
        return self.buffer.memory_bytes()
