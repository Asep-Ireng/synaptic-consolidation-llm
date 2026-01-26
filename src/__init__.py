"""
Synaptic Consolidation for Large Language Models

A biologically-inspired memory consolidation system for continual learning.
"""

from .model import LivingLLM
from .sleep import dream_cycle, synaptic_pruning
from .memory import merge_stm_to_ltm, extract_memories
from .utils import register_activation_hooks, monitor_plasticity

__version__ = "0.1.0"
__all__ = [
    "LivingLLM",
    "dream_cycle",
    "synaptic_pruning",
    "merge_stm_to_ltm",
    "extract_memories",
    "register_activation_hooks",
    "monitor_plasticity",
]
