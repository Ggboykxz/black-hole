"""Black Hole package.

A large language model built from scratch: architecture, tokenizer, data pipeline,
pretraining loop, inference and evaluation.
"""

from .config import BlackHoleConfig, TrainConfig
from .model import BlackHole

__version__ = "0.1.0"
__all__ = ["BlackHole", "BlackHoleConfig", "TrainConfig", "__version__"]
