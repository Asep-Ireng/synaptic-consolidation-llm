"""
Visualization module to generate graphs from actual training metrics.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

# Set up a clean, scientific styling for when you export graphs
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def plot_loss_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    sleep_epochs: Optional[List[int]] = None,
    save_path: str = "experiments/plots/real_loss_curves.png"
):
    """
    Plot actual cross-entropy loss over training epochs.
    Shows the effect of sleep (consolidation) on validation loss if provided.
    """
    ensure_dir(save_path)
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses and len(val_losses) == len(train_losses):
        plt.plot(epochs, val_losses, 'r--', label='Validation Loss', linewidth=2)
        
    if sleep_epochs:
        for epoch in sleep_epochs:
            plt.axvline(x=epoch, color='gray', linestyle=':', alpha=0.7, 
                        label='Sleep Phase (Consolidation)' if epoch == sleep_epochs[0] else "")
            
    plt.title('Training & Validation Loss (Actual Run)', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_plasticity_vs_stability(
    sleep_cycles: List[int],
    plasticity_scores: List[float],
    stability_scores: List[float],
    save_path: str = "experiments/plots/real_plasticity_stability.png"
):
    """
    Plot actual measured plasticity vs stability across consolidation cycles.
    """
    ensure_dir(save_path)
    plt.figure(figsize=(10, 6))
    
    plt.plot(sleep_cycles, stability_scores, 'purple', marker='s', label='Cortical Stability', linewidth=2)
    plt.plot(sleep_cycles, plasticity_scores, '#2ca02c', marker='o', label='Hippocampal Plasticity', linewidth=2)
    
    plt.title('Measured Plasticity vs Stability Tracking', fontsize=14)
    plt.xlabel('Sleep Cycles Completed', fontsize=12)
    plt.ylabel('Metric Score', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_forgetting_curve(
    time_steps: List[int],
    retention_scores: List[float],
    baseline_retention: Optional[List[float]] = None,
    save_path: str = "experiments/plots/real_forgetting_curve.png"
):
    """
    Plot memory retention of a specific learned fact over sequential new tasks.
    """
    ensure_dir(save_path)
    plt.figure(figsize=(10, 6))
    
    plt.plot(time_steps, retention_scores, 'b-', label='LivingLLM (With Sleep)', linewidth=2.5)
    
    if baseline_retention and len(baseline_retention) == len(time_steps):
        plt.plot(time_steps, baseline_retention, 'r--', label='Standard PEFT (Without Sleep)', linewidth=2)
        
    plt.title('Memory Retention Tracking', fontsize=14)
    plt.xlabel('Interference Events (Time)', fontsize=12)
    plt.ylabel('Retention / Accuracy (%)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
