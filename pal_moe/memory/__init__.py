"""Memory: the v3 FAST path (`FastMemory`) and the v1 prototype memory.

`prototype_memory` and `generative` are v1 modules; they live in
`pal_moe.legacy.memory` and the names below are aliases of the same objects.
"""

from .kv_store import FastMemory, MemoryHit
from .prototype_memory import Prototype, PrototypeMemory

__all__ = ["FastMemory", "MemoryHit", "Prototype", "PrototypeMemory"]
