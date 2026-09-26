"""The v3 router interface and the purity guard.

A router maps a frozen key `z` to one score per expert (task) and reads only
frozen keys and registered buffers. Its trainable parameter count is zero, and
that is asserted at runtime on every write / forget / consolidate / predict
(guard 4, "router purity").
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


class RouterPurityError(AssertionError):
    pass


@runtime_checkable
class TaskRouter(Protocol):
    name: str

    def scores(self, z: torch.Tensor) -> torch.Tensor:  # [B, T]
        ...

    def top_k(
        self, z: torch.Tensor, k: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def parameter_count(self) -> int: ...


def trainable_count(obj) -> int:
    """Trainable scalars reachable from a router: nn.Parameters or grad tensors."""
    seen, total = set(), 0
    stack = [obj]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, torch.nn.Module):
            total += sum(p.numel() for p in cur.parameters())
            total += sum(b.numel() for b in cur.buffers() if b.requires_grad)
        elif isinstance(cur, torch.Tensor):
            total += cur.numel() if cur.requires_grad else 0
        elif hasattr(cur, "__dict__") and not isinstance(cur, type):
            stack.extend(
                v
                for v in vars(cur).values()
                if isinstance(v, (torch.nn.Module, torch.Tensor))
                or hasattr(v, "parameter_count")
            )
    return total


def assert_pure(router) -> int:
    n = trainable_count(router)
    if n != 0 or router.parameter_count() != 0:
        raise RouterPurityError(
            f"router {getattr(router, 'name', router)!r} holds {n} trainable scalars"
        )
    return 0
