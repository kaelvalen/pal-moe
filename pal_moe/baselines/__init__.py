from .naive import NaiveFineTuning
from .ewc import EWC
from .replay import ReplayTrainer
from .standard_moe import StandardMoE
from .der import DERPP, ERACE
from .agem import AGEM
from .icarl import ICaRL

__all__ = [
    "NaiveFineTuning",
    "EWC",
    "ReplayTrainer",
    "StandardMoE",
    "DERPP",
    "ERACE",
    "AGEM",
]
