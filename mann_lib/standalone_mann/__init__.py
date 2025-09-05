"""
Standalone MANN Library
Memory-Augmented Neural Network implementation độc lập
"""

from .mann_core import MemoryAugmentedNetwork, MemoryInterface, MemoryBankEntry
from .mann_api import MANNClient
from .mann_config import MANNConfig

__version__ = "1.0.0"
__all__ = [
    "MemoryAugmentedNetwork",
    "MemoryInterface", 
    "MemoryBankEntry",
    "MANNClient",
    "MANNConfig"
]
