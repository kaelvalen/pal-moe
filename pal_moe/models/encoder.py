"""
Shared Encoder and EMA Encoder modules for PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts).

Provides frozen or EMA representations to avoid representation drift during continual learning.
"""

import copy
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedEncoder(nn.Module):
    """
    Shared representation extractor h(x) that maps raw inputs into feature space R^d.
    Supports MLP (for MNIST/tabular) and ConvNet (for CIFAR/images).
    """

    RESNET_ARCHES = ("resnet18", "resnet34", "resnet50")
    # Vision transformers (torchvision, ImageNet weights). They require 224x224
    # inputs, so the encoder resizes internally and maps the dataset
    # normalization back to ImageNet statistics (see `_vit_features`).
    VIT_ARCHES = ("vit_b_16", "vit_b_32", "vit_l_16")
    VIT_WEIGHT_ENUMS = {
        "vit_b_16": "ViT_B_16_Weights",
        "vit_b_32": "ViT_B_32_Weights",
        "vit_l_16": "ViT_L_16_Weights",
    }
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (256, 128),
        output_dim: int = 128,
        arch: str = "mlp",
        dropout: float = 0.0,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        backbone_weights: str = "none",
        input_mean: Optional[tuple[float, ...]] = None,
        input_std: Optional[tuple[float, ...]] = None,
        in_channels: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch
        self.backbone_weights = backbone_weights
        self.conv_channels = tuple(conv_channels)
        # Input channel count for conv/resnet backbones. None keeps the legacy
        # heuristic (3072 flattened dims = RGB CIFAR, everything else = 1
        # channel), so existing runs are unaffected; folder streams pass 3.
        self.in_channels = in_channels
        # Dataset normalisation applied upstream (e.g. the CIFAR mean/std used
        # by the task loaders). Only the ViT preprocessor consumes these: it
        # undoes the dataset normalisation and re-applies ImageNet statistics.
        self.input_mean = tuple(input_mean) if input_mean is not None else None
        self.input_std = tuple(input_std) if input_std is not None else None
        # Backbone output width before the projection head (None for mlp/conv).
        # Used to detect an identity projection (feature_dim == backbone width),
        # which makes the frozen features seed-independent.
        self.backbone_feature_dim: Optional[int] = None

        if arch in self.RESNET_ARCHES:
            self.net = self._build_resnet(arch, input_dim, output_dim, backbone_weights)
        elif arch in self.VIT_ARCHES:
            self.net = self._build_vit(arch, output_dim, backbone_weights)
        elif arch == "mlp":
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, h_dim),
                        nn.LayerNorm(h_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    ]
                )
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.ReLU(inplace=True))
            self.net = nn.Sequential(*layers)
        elif arch == "conv":
            # For 3-channel (CIFAR) or 1-channel (MNIST) inputs.
            # Pools after every conv block except the last, so the default
            # (32, 64, 128) reproduces the original 3-stage architecture.
            in_channels = self.in_channels or (3 if input_dim == 3072 else 1)
            layers = []
            prev_ch = in_channels
            for i, ch in enumerate(conv_channels):
                layers += [
                    nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                ]
                if i < len(conv_channels) - 1:
                    layers.append(nn.MaxPool2d(2))  # 16x16 or 14x14
                prev_ch = ch
            layers += [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(prev_ch, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True),
            ]
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    def _build_resnet(
        self,
        arch: str,
        input_dim: int,
        output_dim: int,
        backbone_weights: str,
    ) -> nn.Sequential:
        """
        Torchvision ResNet backbone with a fresh projection head.

        `backbone_weights="imagenet"` initialises from the ImageNet checkpoint
        (strong frozen features), "none" keeps the random init (pair it with
        SimCLR/AE pretraining). First-layer channels are adapted for 1-channel
        inputs by averaging the pretrained RGB filters.
        """
        import torchvision.models as tvm

        model_fn = getattr(tvm, arch, None)
        if model_fn is None:
            raise ValueError(f"torchvision has no model {arch!r}")
        weights = None
        if backbone_weights == "imagenet":
            weight_enum = getattr(
                tvm, arch.replace("resnet", "ResNet") + "_Weights", None
            )
            if weight_enum is None:
                raise ValueError(f"no ImageNet weights for {arch!r}")
            weights = weight_enum.IMAGENET1K_V1
        backbone = model_fn(weights=weights)

        in_channels = self.in_channels or (3 if input_dim == 3072 else 1)
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            if weights is not None:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            backbone.conv1 = new_conv

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return nn.Sequential(
            backbone,
            nn.Linear(feature_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def _build_vit(
        self,
        arch: str,
        output_dim: int,
        backbone_weights: str,
    ) -> nn.Sequential:
        """
        Torchvision ViT backbone with a fresh projection head.

        Frozen ImageNet ViTs are the strongest representation available without
        extra dependencies. Inputs are resized to the checkpoint's native
        resolution (224) and re-normalised to ImageNet statistics inside
        `_vit_features`; when ``output_dim`` equals the backbone width the head
        is the identity, so the frozen features are seed-independent and a
        persistent feature cache can be shared across seeds.
        """
        import torchvision.models as tvm

        model_fn = getattr(tvm, arch, None)
        if model_fn is None:
            raise ValueError(f"torchvision has no model {arch!r}")
        weights = None
        if backbone_weights == "imagenet":
            enum_name = self.VIT_WEIGHT_ENUMS.get(arch)
            weight_enum = getattr(tvm, enum_name, None) if enum_name else None
            if weight_enum is None:
                raise ValueError(f"no ImageNet weights for {arch!r}")
            weights = weight_enum.IMAGENET1K_V1
        backbone = model_fn(weights=weights)

        self._vit_input_size = 224
        if weights is not None:
            try:
                crop = weights.transforms().crop_size
                if isinstance(crop, (list, tuple)):
                    crop = crop[-1]
                self._vit_input_size = int(crop)
            except Exception:  # pragma: no cover - defensive, weights differ by version
                pass

        feature_dim = backbone.heads.head.in_features
        self.backbone_feature_dim = int(feature_dim)
        backbone.heads.head = nn.Identity()
        if output_dim == feature_dim:
            head: nn.Module = nn.Identity()
        else:
            head = nn.Sequential(
                nn.Linear(feature_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True),
            )
        return nn.Sequential(backbone, head)

    def _vit_features(self, x: torch.Tensor) -> torch.Tensor:
        """Resize + re-normalise inputs to ImageNet statistics, then run the ViT."""
        if x.dim() == 3:  # unbatched input
            x = x.unsqueeze(0)
        if x.size(1) == 1:  # grayscale -> RGB by channel repetition
            x = x.repeat(1, 3, 1, 1)
        size = getattr(self, "_vit_input_size", 224)
        if x.size(-1) != size or x.size(-2) != size:
            x = F.interpolate(
                x,
                size=(size, size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if self.input_mean is not None and self.input_std is not None:
            mean = x.new_tensor(self.input_mean).view(1, -1, 1, 1)
            std = x.new_tensor(self.input_std).view(1, -1, 1, 1)
            x = x * std + mean  # undo the dataset normalisation -> [0, 1]
        mean = x.new_tensor(self.IMAGENET_MEAN).view(1, -1, 1, 1)
        std = x.new_tensor(self.IMAGENET_STD).view(1, -1, 1, 1)
        x = (x - mean) / std
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch in self.VIT_ARCHES:
            return self._vit_features(x)
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

    def train(self, mode: bool = True) -> "SharedEncoder":
        """
        A frozen encoder ignores ``train()`` entirely (it stays in eval mode), so
        BatchNorm running statistics cannot drift when a wrapper calls
        ``model.train()``. Measured artifact: the baselines' replay features
        drifted this way, costing DER++ ~11 accuracy points on Split-CIFAR-10.
        """
        if getattr(self, "_frozen", False):
            super().train(False)
            return self
        super().train(mode)
        return self

    def freeze(self) -> None:
        self._frozen = True
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        self._frozen = False
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
        optimizer = torch.optim.Adam(
            list(self.parameters()) + list(decoder.parameters()), lr=lr
        )

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

    def pretrain_contrastive(
        self,
        dataloader: Any,
        device: torch.device = torch.device("cpu"),
        epochs: int = 2,
        lr: float = 1e-3,
        temperature: float = 0.1,
        augment_fn: Optional[callable] = None,
    ) -> float:
        """
        Self-supervised contrastive pretraining (SimCLR-style InfoNCE).
        Forces representations of similar instances together while separating distinct instances,
        producing highly discriminative, cluster-separated latent representations.
        """
        proj_head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, 64),
        ).to(device)

        self.to(device)
        self.train()
        proj_head.train()
        optimizer = torch.optim.Adam(
            list(self.parameters()) + list(proj_head.parameters()), lr=lr
        )

        total_loss = 0.0
        n_batches = 0

        for _ in range(epochs):
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(device)
                b = x.size(0)
                if b < 2:
                    continue

                # Generate two stochastic augmentations
                # 1) Additive noise & scaling
                if augment_fn is not None:
                    x1 = augment_fn(x)
                    x2 = augment_fn(x)
                else:
                    noise1 = torch.randn_like(x) * 0.08
                    noise2 = torch.randn_like(x) * 0.08
                    scale1 = (
                        torch.empty(b, 1, device=device).uniform_(0.85, 1.15)
                        if x.dim() == 2
                        else torch.empty(b, 1, 1, 1, device=device).uniform_(0.85, 1.15)
                    )
                    scale2 = (
                        torch.empty(b, 1, device=device).uniform_(0.85, 1.15)
                        if x.dim() == 2
                        else torch.empty(b, 1, 1, 1, device=device).uniform_(0.85, 1.15)
                    )

                    x1 = torch.clamp(x * scale1 + noise1, -3.0, 3.0)
                    x2 = torch.clamp(x * scale2 + noise2, -3.0, 3.0)

                optimizer.zero_grad()
                z1 = F.normalize(proj_head(self(x1)), dim=-1)
                z2 = F.normalize(proj_head(self(x2)), dim=-1)

                z = torch.cat([z1, z2], dim=0)  # [2b, 64]
                sim = torch.mm(z, z.t()) / temperature  # [2b, 2b]
                mask = torch.eye(2 * b, dtype=torch.bool, device=device)
                sim.masked_fill_(mask, -1e9)

                labels = torch.cat(
                    [
                        torch.arange(b, 2 * b, device=device),
                        torch.arange(0, b, device=device),
                    ],
                    dim=0,
                )

                loss = F.cross_entropy(sim, labels)
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

    Note: the benchmark runner currently constructs every DynamicMoE with
    ``use_ema_encoder=False`` (the frozen-encoder recipe makes the EMA copy
    redundant), so this path is not exercised by the published results. It is
    kept for online-encoder experiments; `PrototypeMemory.refresh_representations`
    accepts either the EMA or the online encoder.
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
            ema_param.data.mul_(self.decay).add_(
                online_param.data, alpha=1.0 - self.decay
            )
        # Also copy running stats for batchnorm
        for ema_buf, online_buf in zip(
            self.ema_model.buffers(), online_encoder.buffers()
        ):
            ema_buf.copy_(online_buf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.ema_model(x)
