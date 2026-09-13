"""
PAL-MoE: Prototype-Anchored Lifelong Mixture of Experts in PyTorch.
Continual learning framework for mitigating catastrophic forgetting in dynamic MoE.
"""


from .models.moe import DynamicMoE, PALMoE
from .models.encoder import SharedEncoder, EMAEncoder
from .models.router import DynamicRouter
from .models.expert import MLPExpert
from .memory.prototype_memory import PrototypeMemory
from .trigger.expert_trigger import QuantitativeTrigger
from .builder.expert_builder import ExpertBuilder
from .adaptation.ttt import ContinualTrainer, TestTimeAdapter
from .evaluation.metrics import ContinualEvaluator

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
