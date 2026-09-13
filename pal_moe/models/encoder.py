"""
Shared Encoder and EMA Encoder modules for PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts).

Provides frozen or EMA representations to avoid representation drift during continual learning.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Any


class SharedEncoder(nn.Module):
    """
    Shared representation extractor h(x) that maps raw inputs into feature space R^d.
    Supports MLP (for MNIST/tabular) and ConvNet (for CIFAR/images).
    """
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (256, 128),
        output_dim: int = 128,
        arch: Literal["mlp", "conv"] = "mlp",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch

        if arch == "mlp":
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ])
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.ReLU(inplace=True))
            self.net = nn.Sequential(*layers)
        elif arch == "conv":
            # For 3-channel (CIFAR) or 1-channel (MNIST) inputs
            in_channels = 3 if input_dim == 3072 else 1
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 16x16 or 14x14
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 8x8 or 7x7
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "mlp":
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            return self.net(x)
        else:
            if x.dim() == 2:
                # reshape if flattened
                if self.input_dim == 784:
                    x = x.view(-1, 1, 28, 28)
                elif self.input_dim == 3072:
                    x = x.view(-1, 3, 32, 32)
            return self.net(x)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def pretrain_unsupervised(
        self,
        dataloader: Any,
        device: torch.device = torch.device("cpu"),
        epochs: int = 1,
        lr: float = 2e-3,
    ) -> float:
        """
        Unsupervised / self-supervised pre-training via reconstruction.
        Learns domain representations without using task labels.
        """
        decoder = nn.Sequential(
            nn.Linear(self.output_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.input_dim),
        ).to(device)

        self.to(device)
        self.train()
        decoder.train()
        optimizer = torch.optim.Adam(list(self.parameters()) + list(decoder.parameters()), lr=lr)

        total_loss = 0.0
        n_batches = 0

        for _ in range(epochs):
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                x = x.to(device)

                optimizer.zero_grad()
                h = self(x)
                rec = decoder(h)
                loss = F.mse_loss(rec, x)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        self.eval()
        return total_loss / max(n_batches, 1)


class EMAEncoder(nn.Module):
    """
    Maintains an exponential moving average (EMA) copy of the encoder:
        theta_ema = beta * theta_ema + (1 - beta) * theta_online
    This ensures representation stability for prototype memory and routing features.
    """
    def __init__(self, encoder: SharedEncoder, decay: float = 0.99):
        super().__init__()
        self.decay = decay
        self.ema_model = copy.deepcopy(encoder)
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_model.eval()

    @torch.no_grad()
    def update(self, online_encoder: SharedEncoder) -> None:
        for ema_param, online_param in zip(
            self.ema_model.parameters(), online_encoder.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(online_param.data, alpha=1.0 - self.decay)
        # Also copy running stats for batchnorm
        for ema_buf, online_buf in zip(
            self.ema_model.buffers(), online_encoder.buffers()
        ):
            ema_buf.copy_(online_buf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.ema_model(x)
