"""
PAL-MoE: Prototype-Anchored Lifelong Mixture of Experts in PyTorch.
Continual learning framework for mitigating catastrophic forgetting in dynamic MoE.
"""

from .legacy.adaptation.ttt import ContinualTrainer, TestTimeAdapter
from .legacy.builder.expert_builder import ExpertBuilder
from .eval.metrics import ContinualEvaluator
from .legacy.memory.prototype_memory import PrototypeMemory
from .legacy.models.encoder import EMAEncoder, SharedEncoder
from .legacy.models.expert import MLPExpert
from .legacy.models.moe import DynamicMoE, PALMoE
from .legacy.models.router import DynamicRouter
from .legacy.trigger.expert_trigger import QuantitativeTrigger

__version__ = "0.1.0"

__all__ = [
    "PALMoE",
    "DynamicMoE",
    "SharedEncoder",
    "EMAEncoder",
    "DynamicRouter",
    "MLPExpert",
    "PrototypeMemory",
    "QuantitativeTrigger",
    "ExpertBuilder",
    "ContinualTrainer",
    "TestTimeAdapter",
    "ContinualEvaluator",
]
