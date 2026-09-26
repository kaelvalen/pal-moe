"""v3 core: frozen backbone adapters and state hashing.

- `features`       cached frozen-feature tasks (moved from `experiments/s2_ladder.py`)
- `constructions`  S6b `coherent` / `dispersed` task constructions (moved from s6b)
- `hashing`        bitwise content hashes for the state / reversibility guards
- `backbones`      `FrozenFeatureBackbone` (cached features, identity keys) and the
                   torchvision / legacy encoders through `pal_moe.arch.backbones`
- `hf_lm`          HuggingFace causal-LM adapter with hidden-state hooks (lazy import:
                   `transformers` is only needed when it is used)
"""

from .backbones import FrozenFeatureBackbone
from .hashing import digest, file_digest, module_digest, tensor_digest

__all__ = [
    "FrozenFeatureBackbone",
    "digest",
    "file_digest",
    "module_digest",
    "tensor_digest",
]
