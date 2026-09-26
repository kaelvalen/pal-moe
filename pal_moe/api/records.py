"""Records returned by the v3 API: edits, state hashes, predictions, reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Example:
    """One item for the FAST path: a single input and its value (label / target)."""

    x: Any
    y: Any = None


@dataclass
class Batch:
    """A batch for the MEDIUM path. `task` groups batches for `by_arrival`."""

    x: Any
    y: Any
    task: int | None = None


@dataclass(frozen=True)
class StateHash:
    """`(base_hash, ordered edit log)` and one digest over both."""

    base_hash: str
    edits: tuple  # ((edit_id, kind, content_hash), ...) in arrival order
    digest: str


@dataclass
class EditRecord:
    id: str
    kind: str  # "fast" | "medium" | "consolidation"
    content_hash: str
    order_hash: str = ""  # digest of the live medium-path edit order after this edit
    locality_report: dict = field(default_factory=dict)
    reversibility_report: dict = field(default_factory=dict)
    order_report: dict = field(default_factory=dict)
    purity_report: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


@dataclass
class ConsolidationReport:
    record: EditRecord
    policy: str
    groups: list[list[int]]
    experts_added: int
    frozen_parameters: int


@dataclass
class Prediction:
    labels: torch.Tensor  # [B] final answer
    logits: torch.Tensor  # [B, C] the parametric path's logits (before memory override)
    expert_ids: torch.Tensor | None  # [B, k] routed experts, best first
    expert_scores: torch.Tensor | None  # [B, k]
    source: list[str]  # per sample: "memory" | "experts" | "medium"
    memory_hits: list[list[Any]] = field(default_factory=list)
    medium_labels: torch.Tensor | None = None  # the medium readout alone (ridge_alone)


class GuardViolation(RuntimeError):
    def __init__(self, guard: str, report: dict):
        super().__init__(f"guard {guard!r} violated: {report}")
        self.guard, self.report = guard, report


class ReversibilityError(GuardViolation):
    pass
