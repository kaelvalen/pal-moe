"""
PAL-MoE: Prototype-Anchored Lifelong Mixture of Experts in PyTorch.
Continual learning framework for mitigating catastrophic forgetting in dynamic MoE.
"""

from .adaptation.ttt import ContinualTrainer, TestTimeAdapter
from .builder.expert_builder import ExpertBuilder
from .evaluation.metrics import ContinualEvaluator
from .memory.prototype_memory import PrototypeMemory
from .models.encoder import EMAEncoder, SharedEncoder
from .models.expert import MLPExpert
from .models.moe import DynamicMoE, PALMoE
from .models.router import DynamicRouter
from .trigger.expert_trigger import QuantitativeTrigger

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
