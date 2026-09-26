"""Readouts: the existing S1 registry (`ncm`, `cosine`, `linear`, `logistic`, `ridge`, `mlp`).

The implementations stay in `pal_moe/arch/readouts.py`: `pal_moe.arch` registers them
at import time through `pal_moe.arch.registry`, and moving the file would put a
package-init cycle between `arch` and `readout`. This package is the v3 name for the
same objects (identity, not copies): `pal_moe.readout.RidgeReadout is
pal_moe.arch.RidgeReadout`.

For the v3 medium path use `pal_moe.edit.LinearStats` (float64, subtractable); the
float32 `RidgeReadout` is kept for bitwise reproduction of every stored result.
"""

from pal_moe.arch.readouts import (
    CosineReadout,
    LinearReadout,
    LogisticReadout,
    MLPReadout,
    NCMReadout,
    RidgeReadout,
    mask_unseen,
)
from pal_moe.arch.registry import READOUTS, build_readout

__all__ = [
    "READOUTS",
    "CosineReadout",
    "LinearReadout",
    "LogisticReadout",
    "MLPReadout",
    "NCMReadout",
    "RidgeReadout",
    "build_readout",
    "mask_unseen",
]
