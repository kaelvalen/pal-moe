"""
Latent generative replay.

PAL-MoE's "pure" mode stores only 128/256-d latent vectors. A generator fitted
on those latents can produce unlimited pseudo-exemplars without ever touching
raw data:

- "gaussian": class-balanced diagonal Gaussians (mean/std per class estimated
  from the stored exemplars). Zero training, very stable, works from a handful
  of exemplars per class.
- "vae": a small class-conditional VAE over the latent exemplars. Slower to fit
  but samples new combinations instead of jittering class means.

Both return (features, labels) batches on the requested device.
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LatentReplayGenerator"]

GENERATIVE_MODES = ("gaussian", "vae")


class _ConditionalVAE(nn.Module):
    def __init__(
        self, feature_dim: int, num_classes: int, hidden: int = 128, z_dim: int = 16
    ):
        super().__init__()
        in_dim = feature_dim + num_classes
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + num_classes, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor):
        h = self.encoder(torch.cat([x, y_onehot], dim=-1))
        mu, logvar = self.mu(h), self.logvar(h)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(torch.cat([z, y_onehot], dim=-1))
        return recon, mu, logvar


class LatentReplayGenerator:
    """Fits on prototype memory and samples synthetic (features, labels)."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        mode: str = "gaussian",
        device: Optional[torch.device] = None,
        vae_steps: int = 300,
        vae_lr: float = 1e-2,
        seed: int = 0,
    ):
        if mode not in GENERATIVE_MODES:
            raise ValueError(f"unknown generative mode {mode!r}")
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.mode = mode
        self.device = device or torch.device("cpu")
        self.vae_steps = vae_steps
        self.vae_lr = vae_lr
        self.seed = seed
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        self._fitted = False
        self._means: Optional[torch.Tensor] = None
        self._stds: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None
        self._vae: Optional[_ConditionalVAE] = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ---------------------------------------------------------------- fitting
    def fit(self, prototype_memory: Any) -> bool:
        """Collects the labeled exemplars (or prototype centres) of the memory."""
        rows, labels = [], []
        for proto in prototype_memory.prototypes:
            if (
                proto.x_p is not None
                and proto.y_p is not None
                and proto.x_p.size(0) > 0
            ):
                n = min(proto.x_p.size(0), proto.y_p.size(0))
                rows.append(proto.x_p[:n].float())
                labels.append(proto.y_p[:n].long())
            elif proto.y_p is not None and proto.y_p.numel() > 0:
                rows.append(proto.v_p.view(1, -1).float())
                labels.append(proto.y_p[:1].long())
        if not rows:
            self._fitted = False
            return False
        x = torch.cat(rows, dim=0)
        y = torch.cat(labels, dim=0)
        keep = y >= 0
        if not bool(keep.any()):
            self._fitted = False
            return False
        x, y = x[keep], y[keep]
        if self.mode == "gaussian":
            with torch.no_grad():
                self._fit_gaussian(x, y)
        else:
            self._fit_vae(x, y)
        self._fitted = True
        return True

    @torch.no_grad()
    def _fit_gaussian(self, x: torch.Tensor, y: torch.Tensor) -> None:
        classes = torch.unique(y)
        means, stds = [], []
        global_std = x.std(dim=0).clamp(min=1e-3)
        for c in classes:
            rows = x[y == c]
            means.append(rows.mean(dim=0))
            stds.append(rows.std(dim=0) if rows.size(0) > 1 else global_std)
        self._means = torch.stack(means, dim=0)
        self._stds = torch.stack(stds, dim=0).clamp(min=1e-3)
        self._labels = classes.long()
        self._vae = None

    def _fit_vae(self, x: torch.Tensor, y: torch.Tensor) -> None:
        vae = _ConditionalVAE(self.feature_dim, self.num_classes).to(self.device)
        opt = torch.optim.Adam(vae.parameters(), lr=self.vae_lr)
        y_oh = F.one_hot(y, self.num_classes).float().to(self.device)
        x = x.to(self.device)
        torch.manual_seed(self.seed)
        for _ in range(self.vae_steps):
            opt.zero_grad()
            recon, mu, logvar = vae(x, y_oh)
            loss = F.mse_loss(recon, x) + 1e-3 * (
                -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
            )
            loss.backward()
            opt.step()
        self._vae = vae.eval()

    # ----------------------------------------------------------------- sampling
    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._fitted:
            raise RuntimeError("LatentReplayGenerator.sample called before fit()")
        if self.mode == "gaussian":
            idx = torch.randint(0, self._means.size(0), (n,), generator=self._gen)
            noise = torch.randn(n, self.feature_dim, generator=self._gen)
            feats = self._means[idx] + self._stds[idx] * noise
            return feats.to(self.device), self._labels[idx].to(self.device)
        y = torch.randint(
            0, self.num_classes, (n,), generator=self._gen, device=self.device
        )
        y_onehot = F.one_hot(y, self.num_classes).float()
        z = torch.randn(n, self._vae.z_dim, device=self.device)
        with torch.no_grad():
            feats = self._vae.decoder(torch.cat([z, y_onehot], dim=-1))
        return feats, y
