"""
iCaRL (Rebuffi et al., 2017) baseline: Incremental Classifier and Representation
Learning.

- Class exemplar sets selected by "herding" (iteratively match the class mean).
- Training: cross-entropy on the new classes + knowledge distillation on old
  classes from the previous model snapshot.
- Classification: nearest-class-mean in feature space, using the exemplars.

Interface matches pal_moe.baselines.replay.ReplayTrainer. The wrapper module's
forward() returns negative squared distances to class means (argmin => class),
so it plugs directly into ContinualEvaluator.
"""

import copy
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICaRLWrapper(nn.Module):
    """Wraps (encoder, expert) and classifies by nearest class mean."""

    def __init__(self, network: nn.Module, num_classes: int = 10):
        super().__init__()
        self.network = network
        self.num_classes = num_classes
        # class means in feature space, filled after each task
        self.register_buffer(
            "class_means", torch.zeros(num_classes, self.feature_dim())
        )

    def feature_dim(self) -> int:
        # Feature space = input of the classification expert's first Linear,
        # i.e. the backbone output dimension (extract_features drops the last
        # child module and returns its input dimension).
        children = list(self.network.named_children())
        last_child = children[-1][1]
        lin_layers = [m for m in last_child.modules() if isinstance(m, nn.Linear)]
        if lin_layers:
            return lin_layers[0].in_features
        raise RuntimeError("no Linear layer found in classification module")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # pass through every module except the last Linear classification layer
        out = x
        named = list(self.network.named_children())
        for _, mod in named[:-1]:
            out = mod(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)  # [B, D]
        dists = -((feats.unsqueeze(1) - self.class_means.unsqueeze(0)) ** 2).sum(
            -1
        )  # [B, C]
        return dists


class ICaRL:
    def __init__(
        self,
        model: nn.Module,
        exemplars_per_class: int = 20,
        num_classes: int = 10,
        lr: float = 1e-3,
        distil_weight: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        self.wrapper = ICaRLWrapper(model, num_classes=num_classes).to(device)
        self.exemplars_per_class = exemplars_per_class
        self.lr = lr
        self.distil_weight = distil_weight
        self.device = device
        self.optimizer = torch.optim.Adam(self.wrapper.parameters(), lr=self.lr)
        self.exemplars: dict[int, list[torch.Tensor]] = {}
        self.seen_classes: list[int] = []
        self.old_model: Optional[ICaRLWrapper] = None

    # ---- exemplar management (herding) ----
    def _compute_class_means(self) -> None:
        with torch.no_grad():
            for c, exs in self.exemplars.items():
                if not exs:
                    continue
                stack = torch.stack([e.to(self.device) for e in exs])
                feats = self.wrapper.extract_features(stack)
                self.wrapper.class_means[c] = feats.mean(dim=0)

    def _herding_select(self, feats: torch.Tensor, k: int) -> list[int]:
        """
        Select k exemplars whose running-mean best matches the class mean.

        Greedy iCaRL herding. Vectorized so each selection is one
        matrix-vector product instead of a Python scan with a host sync per
        candidate: ||s + f_i||^2 = ||s||^2 + 2<s, f_i> + ||f_i||^2 and the
        first term is constant, so argmin reduces to 2<s, f_i> + ||f_i||^2.
        """
        n = min(k, feats.size(0))
        current_sum = torch.zeros(feats.size(1), dtype=feats.dtype, device=feats.device)
        feat_sq = (feats * feats).sum(dim=1)
        available = torch.ones(feats.size(0), dtype=torch.bool, device=feats.device)
        selected: list[int] = []
        for _ in range(n):
            scores = 2.0 * (feats @ current_sum) + feat_sq
            scores = scores.masked_fill(~available, float("inf"))
            best = int(scores.argmin().item())
            selected.append(best)
            available[best] = False
            current_sum = current_sum + feats[best]
        return selected

    def update_exemplars(self, train_loader: Any, current_classes: list[int]) -> None:
        # collect features of ALL current-class samples
        class_feats: dict[int, list[torch.Tensor]] = {c: [] for c in current_classes}
        class_raw: dict[int, list[torch.Tensor]] = {c: [] for c in current_classes}
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True
                )
                feats = self.wrapper.extract_features(x)
                for c in current_classes:
                    mask = y == c
                    if mask.any():
                        class_feats[c].append(feats[mask])
                        class_raw[c].append(x[mask])
        for c in current_classes:
            if not class_feats[c]:
                continue
            feats_c = torch.cat(class_feats[c], dim=0)
            raw_c = torch.cat(class_raw[c], dim=0)
            idx = self._herding_select(feats_c, self.exemplars_per_class)
            # keep last exemplars_per_class only (iCaRL stores k per class)
            self.exemplars[c] = [raw_c[i].detach().cpu() for i in idx]
            self.seen_classes = sorted(self.exemplars.keys())
        self._compute_class_means()

    # ---- training ----
    def train_task(
        self,
        task_id: int,
        train_loader: Any,
        epochs: int = 5,
        current_classes: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        if current_classes is None:
            current_classes = [2 * task_id, 2 * task_id + 1]
        self.seen_classes = sorted(set(self.seen_classes + current_classes))
        self.wrapper.train()
        total_loss = torch.zeros((), device=self.device)
        n_updates = 0

        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True
                )
                self.optimizer.zero_grad()

                logits = self.wrapper.network(x)  # full logits [B, C]
                ce = F.cross_entropy(logits, y)

                distil = torch.tensor(0.0, device=self.device)
                if self.old_model is not None:
                    with torch.no_grad():
                        old_logits = self.old_model.network(x)
                    old_classes = [
                        c for c in self.seen_classes if c not in current_classes
                    ]
                    if old_classes:
                        # knowledge distillation via sigmoid BCE over old classes
                        t_old = torch.sigmoid(old_logits[:, old_classes] / 2.0)
                        s_old = torch.sigmoid(logits[:, old_classes] / 2.0)
                        distil = F.binary_cross_entropy(s_old, t_old)

                loss = ce + self.distil_weight * distil
                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach()
                n_updates += 1

        # snapshot model for next task's distillation
        self.old_model = copy.deepcopy(self.wrapper).eval()
        self.update_exemplars(train_loader, current_classes)
        return {
            "task_id": task_id,
            "loss": float(total_loss.item()) / max(n_updates, 1),
        }

    # expose wrapper for evaluation
    def eval_model(self) -> ICaRLWrapper:
        self.wrapper.eval()
        return self.wrapper

    def memory_bytes(self) -> int:
        """Raw exemplars plus the class means (the persistent iCaRL store)."""
        total = 0
        for exemplars in self.exemplars.values():
            for e in exemplars:
                total += e.numel() * e.element_size()
        means = self.wrapper.class_means
        total += means.numel() * means.element_size()
        return int(total)

    def snapshot_bytes(self) -> int:
        """Parameter bytes of the distillation snapshot (`old_model`)."""
        if self.old_model is None:
            return 0
        return int(
            sum(p.numel() * p.element_size() for p in self.old_model.parameters())
        )
