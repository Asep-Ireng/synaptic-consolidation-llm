"""
Utility Functions: Activation Monitoring and Plasticity Tracking
Optimized for 16GB VRAM (RTX 5070 Ti)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ActivationMonitor:
    """
    Hook-based activation monitoring for tracking neural plasticity.
    Memory Optimized: Stores scalar intensity values instead of full tensors.
    """
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def register_hook(self, module: nn.Module, name: str):
        """Register a forward hook to capture activation intensity."""
        
        def hook_fn(module, input, output):
            # MEMORY FIX: Don't store the whole tensor! 
            # Calculate the "Loudness" (L2 Norm) immediately and discard the heavy tensor.
            # output shape: (batch, seq, hidden) -> scalar
            with torch.no_grad():
                intensity = output.detach().float().norm(dim=-1).mean().item()
                
            # If we are processing multiple batches/tokens, we accumulate or overwrite
            # For "Dreaming", usually we just want the latest or average.
            # Here we store the max activation seen in this pass to catch "spikes".
            if name not in self.activations:
                self.activations[name] = intensity
            else:
                self.activations[name] = max(self.activations[name], intensity)
        
        hook = module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook
    
    def clear_hooks(self):
        """Remove all registered hooks and clear data."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def get_activation_stats(self) -> Dict[str, float]:
        """Returns the captured intensity map."""
        return self.activations

def register_activation_hooks(model, layer_names: List[str] = None) -> ActivationMonitor:
    """
    Register activation monitoring hooks on specified layers.
    If layer_names is None, monitors all LoRA layers.
    """
    monitor = ActivationMonitor()
    
    # We iterate through the modules. 
    # NOTE: Adjust 'model.stm_model' depending on your specific class structure.
    # If using PEFT directly, it might be 'model.base_model' or just 'model'
    target_model = getattr(model, "model", model) 
    
    count = 0
    for name, module in target_model.named_modules():
        # Auto-detect LoRA layers if no specific names provided
        is_target = False
        if layer_names:
            is_target = any(ln in name for ln in layer_names)
        elif "lora" in name and isinstance(module, torch.nn.Linear):
            is_target = True
            
        if is_target:
            monitor.register_hook(module, name)
            count += 1
            
    logger.info(f"📊 Monitor attached to {count} layers.")
    return monitor

def monitor_plasticity(model, baseline_weights: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Calculate plasticity by comparing current weights to baseline.
    PERFORMANCE FIX: Calculations are offloaded to CPU to save VRAM.
    """
    plasticity = {}
    
    target_model = getattr(model, "model", model)

    with torch.no_grad():
        for name, param in target_model.named_parameters():
            if "lora" in name and name in baseline_weights:
                # Move current param to CPU for the math, then discard
                current_cpu = param.detach().cpu()
                baseline_cpu = baseline_weights[name] # Assumed to be on CPU already
                
                weight_change = (current_cpu - baseline_cpu).abs().mean().item()
                plasticity[name] = weight_change
    
    return plasticity

def calculate_catastrophic_forgetting(
    model,
    old_prompts: List[str],
    baseline_perplexity: Dict[str, float],
) -> Dict[str, float]:
    """
    Measure catastrophic forgetting by comparing perplexity changes.
    """
    forgetting_scores = {}
    model.model.eval()
    
    with torch.no_grad():
        for prompt in old_prompts:
            inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device)
            # Calculate Loss (Surprise)
            outputs = model.model(**inputs, labels=inputs["input_ids"])
            
            # Perplexity = exp(loss)
            current_ppl = torch.exp(outputs.loss).item()
            
            # Use get() with default to current_ppl to avoid errors if baseline missing
            baseline_ppl = baseline_perplexity.get(prompt, current_ppl)
            
            # Forgetting = How much higher is the perplexity now?
            forgetting_scores[prompt] = max(0, current_ppl - baseline_ppl)
    
    return forgetting_scores

def save_checkpoint(
    model, 
    path: str, 
    optimizer: torch.optim.Optimizer = None, # ADDED: Optimizer state
    metadata: Dict = None
):
    """
    Save full model checkpoint with metadata and optimizer state.
    """
    target_model = getattr(model, "model", model)
    
    checkpoint = {
        "model_state": target_model.state_dict(),
        "sleep_cycles": getattr(model, "sleep_cycles", 0),
        "metadata": metadata or {},
    }
    
    # Save optimizer if provided (Critical for resuming training)
    if optimizer:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    
    torch.save(checkpoint, path)
    logger.info(f"💾 Checkpoint saved: {path}")

def load_checkpoint(
    model, 
    path: str, 
    optimizer: torch.optim.Optimizer = None
) -> Dict:
    """
    Load model checkpoint and restore optimizer if provided.
    """
    checkpoint = torch.load(path, map_location="cpu") # Load to CPU first to manage VRAM
    
    target_model = getattr(model, "model", model)
    target_model.load_state_dict(checkpoint["model_state"])
    
    if hasattr(model, "sleep_cycles"):
        model.sleep_cycles = checkpoint.get("sleep_cycles", 0)
    
    # Restore optimizer
    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.info("🔧 Optimizer state restored.")
    
    logger.info(f"📂 Checkpoint loaded: {path}")
    return checkpoint["metadata"]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Utils module loaded (VRAM Optimized).")