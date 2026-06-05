"""
Synaptic Consolidation for Large Language Models

A biologically-inspired memory consolidation system for continual learning.
"""

from .model import LivingLLM
from .sleep import dream_cycle, synaptic_pruning, full_sleep_cycle
from .memory import merge_stm_to_ltm, extract_memories, reset_stm, memory_stability_score
from .utils import register_activation_hooks, monitor_plasticity
from .visualization import plot_loss_curves, plot_plasticity_vs_stability, plot_forgetting_curve

__version__ = "0.1.0"
__all__ = [
    "LivingLLM",
    "dream_cycle",
    "synaptic_pruning",
    "full_sleep_cycle",
    "merge_stm_to_ltm",
    "extract_memories",
    "reset_stm",
    "memory_stability_score",
    "register_activation_hooks",
    "monitor_plasticity",
    "plot_loss_curves",
    "plot_plasticity_vs_stability",
    "plot_forgetting_curve",
]
