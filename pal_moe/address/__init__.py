"""The fixed address space: frozen keys and parameter-free retrieval over them.

Keys are never trained (C1, AC1, AC3). A key is whatever the frozen backbone emits
at the chosen layer: `FrozenFeatureBackbone.encode` for cached vision features,
`HFCausalLM.keys` for a language model.
"""

from .index import ExactCosineIndex, SearchResult

__all__ = ["ExactCosineIndex", "SearchResult"]
