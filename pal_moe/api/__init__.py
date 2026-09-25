"""The v3 public API: learning is an API call, not a training run.

model = PalMoE(dim=768, num_classes=100, router="ridge_class", canary=canary_feats)
rec = model.write(Batch(z, y, task=0))     # MEDIUM: closed-form, float64, reversible
rec = model.write(Example(z1, 7))          # FAST: one memory row
model.forget(rec.id)                       # exact
model.consolidate("by_arrival")            # SLOW: frozen experts
model.predict(z_test)                      # routed expert ids + scores
model.state()                              # (base_hash, ordered edit log)
"""

from .facade import PalMoE
from .guards import GuardConfig, GuardedEditor
from .records import (
    Batch,
    ConsolidationReport,
    EditRecord,
    Example,
    GuardViolation,
    Prediction,
    ReversibilityError,
    StateHash,
)

__all__ = [
    "Batch",
    "ConsolidationReport",
    "EditRecord",
    "Example",
    "GuardConfig",
    "GuardViolation",
    "GuardedEditor",
    "PalMoE",
    "Prediction",
    "ReversibilityError",
    "StateHash",
]
