"""SLOW path: representation experts and consolidation policies.

- `RepresentationExpert` implementations (identity at init, function-preserving) come
  from the S1 contract in `pal_moe.arch.experts`.
- `ladder` is the Stage 1 expert-bank model (moved verbatim from the runners).
- `policies` decides which data trains which expert: `by_arrival` (control) and the
  gated, experimental `by_confusion`.
"""

from pal_moe.arch.experts import IdentityExpert, ResidualAdapter, ResidualMLPExpert

from .policies import (
    GATED,
    POLICIES,
    PolicyGateError,
    by_arrival,
    by_confusion,
    check_gate,
)

__all__ = [
    "GATED",
    "POLICIES",
    "IdentityExpert",
    "PolicyGateError",
    "ResidualAdapter",
    "ResidualMLPExpert",
    "by_arrival",
    "by_confusion",
    "check_gate",
]
