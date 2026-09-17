"""
Expert merging utilities.

A growing expert pool keeps accuracy but costs latency and memory. These
training-free mergers fold a set of experts into a single one, which is what a
production serving path wants (one model, bounded cost) and what the
capacity-control path can use instead of parameter averaging.

- `model_soup`: uniform parameter average (checkpoint-soup style).
- `ties_merge`: trim each delta to its top-k magnitude, resolve sign conflicts
  by majority, then average the surviving deltas.
- `task_arithmetic`: base + lambda * sum_i (expert_i - base).
"""

import copy
from collections.abc import Sequence
from typing import Optional

import torch

from .models.expert import MLPExpert

__all__ = ["model_soup", "ties_merge", "task_arithmetic"]


def _tensor_names(expert: MLPExpert) -> list[str]:
    return [name for name, _ in expert.named_parameters()]


def model_soup(experts: Sequence[MLPExpert]) -> MLPExpert:
    """Uniform parameter average of the experts (a new expert is returned)."""
    if not experts:
        raise ValueError("model_soup needs at least one expert")
    merged = copy.deepcopy(experts[0])
    with torch.no_grad():
        for name in _tensor_names(merged):
            stacked = torch.stack(
                [dict(expert.named_parameters())[name].detach() for expert in experts]
            )
            dict(merged.named_parameters())[name].copy_(stacked.mean(dim=0))
    return merged


def ties_merge(
    experts: Sequence[MLPExpert],
    base: Optional[MLPExpert] = None,
    top_k: float = 0.2,
) -> MLPExpert:
    """
    TIES-style merge of `experts` relative to `base` (defaults to the first
    expert). Keeps the top-k fraction of each delta by magnitude, elects the
    majority sign per coordinate, and averages the survivors.
    """
    if not experts:
        raise ValueError("ties_merge needs at least one expert")
    base = base or experts[0]
    base_params = dict(base.named_parameters())
    merged = copy.deepcopy(base)
    merged_params = dict(merged.named_parameters())
    with torch.no_grad():
        for name in _tensor_names(merged):
            deltas = [
                dict(e.named_parameters())[name].detach() - base_params[name]
                for e in experts
            ]
            trim = []
            for delta in deltas:
                k = max(1, int(delta.numel() * top_k))
                flat = delta.flatten()
                keep = flat.abs().topk(k).indices
                mask = torch.zeros_like(flat)
                mask[keep] = 1.0
                trim.append((flat * mask).view_as(delta))
            stacked = torch.stack(trim)
            sign = torch.sign(stacked.sum(dim=0))
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            masked = stacked * (torch.sign(stacked) == sign)
            merged_params[name].copy_(
                base_params[name]
                + masked.sum(dim=0) / masked.ne(0).sum(dim=0).clamp(min=1)
            )
    return merged


def task_arithmetic(
    experts: Sequence[MLPExpert], base: Optional[MLPExpert] = None, scaling: float = 1.0
) -> MLPExpert:
    """base + scaling * sum_i (expert_i - base)."""
    if not experts:
        raise ValueError("task_arithmetic needs at least one expert")
    base = base or experts[0]
    merged = copy.deepcopy(base)
    base_params = dict(base.named_parameters())
    merged_params = dict(merged.named_parameters())
    with torch.no_grad():
        for name in _tensor_names(merged):
            delta = torch.zeros_like(base_params[name])
            for expert in experts:
                delta = delta + (
                    dict(expert.named_parameters())[name].detach() - base_params[name]
                )
            merged_params[name].copy_(base_params[name] + scaling * delta)
    return merged
