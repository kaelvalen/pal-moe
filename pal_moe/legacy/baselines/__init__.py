from .agem import AGEM
from .der import DERPP, ERACE
from .ewc import EWC
from .icarl import ICaRL
from .latent_replay import LatentReplayTrainer
from .mir import MIR
from .naive import NaiveFineTuning
from .replay import ReplayTrainer

__all__ = [
    "NaiveFineTuning",
    "EWC",
    "ReplayTrainer",
    "DERPP",
    "ERACE",
    "AGEM",
    "ICaRL",
    "LatentReplayTrainer",
    "MIR",
]
