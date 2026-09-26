"""Parameter-free routers over the fixed address space.

`prototype` (E0 / AC3) and `ridge_class` (E-TID2) behind one interface; the purity
guard asserts zero trainable parameters at runtime.
"""

from .base import RouterPurityError, TaskRouter, assert_pure, trainable_count
from .prototype import PrototypeTaskRouter
from .ridge_class import RidgeClassRouter

ROUTERS = {"prototype": PrototypeTaskRouter, "ridge_class": RidgeClassRouter}

__all__ = [
    "ROUTERS",
    "PrototypeTaskRouter",
    "RidgeClassRouter",
    "RouterPurityError",
    "TaskRouter",
    "assert_pure",
    "trainable_count",
]
