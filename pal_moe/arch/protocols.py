"""
The S1 architecture contract: four interfaces, one direction of information.

    X --Backbone--> Z --Expert--> Z' --Readout--> Y
                          ^
                          |
                        Router

Nothing in this module changes behaviour. It exists so that a config string can
select a backbone, an expert, a router and a readout independently, and so that
the training loop never has to know which implementation it is holding. The v1
modules (`SharedEncoder`, `MLPExpert`, `DynamicRouter`, `DynamicMoE`) keep
working untouched; the S1 classes wrap or re-expose them.

Why the split is not cosmetic (measured, docs/PALMOE_V2_SPEC.md and E0):

    L2a  one shared expert, trained jointly   76.62   <- capacity is not the problem
    L2b  one shared expert, sequential        55.67   <- sharing under CL is
    L3   one expert per task                  70.56   <- isolation is what the bank buys
    L4   L3 + oracle routing                  97.64   <- and selection is what it costs

An "expert" that is really a classifier head (`Z -> Y`) cannot express that
distinction: it fuses adaptation and readout, so a capacity experiment and a
readout experiment cannot be run separately. Hence two expert protocols.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

__all__ = [
    "Backbone",
    "RepresentationExpert",
    "ClassificationExpert",
    "Readout",
    "Router",
    "is_representation_expert",
    "is_classification_expert",
]


@runtime_checkable
class Backbone(Protocol):
    """`X -> Z`. Frozen or trainable is a property of the instance, not the type."""

    output_dim: int

    def encode(self, x: Tensor) -> Tensor:
        """Map inputs to the stable representation `z` of shape `[B, output_dim]`."""


@runtime_checkable
class RepresentationExpert(Protocol):
    """`Z -> Z`. Adaptation only: this is what an MoE expert should be.

    `transform` must be initializable as the identity, so that adding an expert
    is function-preserving and a bank of N experts starts equivalent to N=1.
    """

    def transform(self, z: Tensor) -> Tensor:
        """Map `z` to an adapted `z'` of the same shape."""


@runtime_checkable
class ClassificationExpert(Protocol):
    """`Z -> Y`. The legacy v1 shape, kept so that v1 does not break.

    A module satisfying this protocol fuses adaptation and readout and must NOT
    be used where the contract requires the two to be separable (the whole point
    of the L2/L3 comparison). It is registered under its own kind so that the
    distinction is visible in configs and in results.
    """

    num_classes: int

    def classify(self, z: Tensor) -> Tensor:
        """Map `z` to class logits of shape `[B, num_classes]`."""


@runtime_checkable
class Readout(Protocol):
    """`Z -> Y`, fittable. NCM, ridge, logistic and an MLP are all readouts.

    A training-free readout (NCM) implements `fit` as registration; the
    signature is the same so that a study can swap one for another without
    touching the loop.
    """

    num_classes: int

    def fit(self, z: Tensor, y: Tensor, seen_classes: list[int] | None = None) -> dict:
        """Update from a batch or a task. Returns a small report dict."""

    def predict(self, z: Tensor) -> Tensor:
        """Class logits `[B, num_classes]`."""


@runtime_checkable
class Router(Protocol):
    """`Z -> distribution over experts`. Kept separate from both expert and readout.

    A router must be able to report a *candidate set* (top-k), because that is
    the quantity the L4 gap is about: E0 measured candidate recall@3 = 88.6%
    against 27 points of oracle headroom.
    """

    num_experts: int

    def route(self, z: Tensor) -> Tensor:
        """Routing probabilities `[B, num_experts]`."""


def is_representation_expert(module: object) -> bool:
    """Structural check that a module is `Z -> Z` and not a fused head."""
    return isinstance(module, RepresentationExpert) and not isinstance(
        module, ClassificationExpert
    )


def is_classification_expert(module: object) -> bool:
    return isinstance(module, ClassificationExpert)


def assert_shapes(module, z: Tensor, name: str) -> Tensor:
    """Shared shape guard used by the contract tests and by the builders."""
    out = module(z)
    if not isinstance(out, Tensor):
        raise TypeError(f"{name} must return a Tensor, got {type(out).__name__}")
    if out.shape[0] != z.shape[0]:
        raise ValueError(f"{name} changed the batch size: {out.shape} vs {z.shape}")
    return out


def identity_check(module, z: torch.Tensor, atol: float = 1e-6) -> bool:
    """True when a freshly built expert is the identity (function-preserving)."""
    with torch.no_grad():
        return bool(torch.allclose(module(z), z, atol=atol))
