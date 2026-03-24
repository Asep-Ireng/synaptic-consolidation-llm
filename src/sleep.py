"""
Sleep Cycle Implementation: Consolidation and Pruning
Optimized for 16GB VRAM (Generative Replay & Adaptive Pruning)
"""

import torch
import random
import logging
from typing import List, Dict, Optional
from .memory import merge_stm_to_ltm, reset_stm, load_ltm_to_stm

logger = logging.getLogger(__name__)

class DreamGenerator:
    """
    Generates synthetic 'dream' data based on recent memories.
    This prevents the need to store infinite history logs in RAM.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate_dream(self, seed_text: str, max_length: int = 64) -> str:
        """
        Hallucinates a continuation of a thought (Generative Replay).
        """
        self.model.eval()
        inputs = self.tokenizer(seed_text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.8, # High temp for "creative" dreams
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def dream_cycle(
    model,
    prompts_to_consolidate: List[str], # Just the short prompts/topics, not full logs
    num_dreams: int = 5,
    learning_rate: float = 1e-5,
) -> Dict[str, float]:
    """
    Simulate sleep consolidation through Generative Replay.
    
    Instead of re-reading exact logs, the model 'dreams' (generates) 
    content based on key topics and reinforces those neural pathways.
    """
    logger.info(f"💤 Starting REM Sleep ({num_dreams} dreams)...")
    
    # Enable training on STM adapters
    model.model.train()
    
    # We use a fresh optimizer for the sleep cycle to avoid messing up wake-state momentum
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.model.parameters()),
        lr=learning_rate
    )
    
    dream_gen = DreamGenerator(model.model, model.tokenizer)
    metrics = {"losses": []}
    
    for i in range(num_dreams):
        # 1. Pick a topic to dream about
        seed = random.choice(prompts_to_consolidate) if prompts_to_consolidate else "The system logic is"
        
        # 2. GENERATE the dream (Self-Supervised Data Creation)
        # We temporarily switch to eval to generate, then back to train
        dream_content = dream_gen.generate_dream(seed)
        model.model.train() # Switch back to train mode
        
        # 3. Train on the dream (Consolidate the pathway)
        inputs = model.tokenizer(
            dream_content,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(model.device)
        
        outputs = model.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # 4. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metrics["losses"].append(loss.item())
        if i % 2 == 0:
            logger.info(f"   Dream {i+1}: '{seed}...' -> Loss: {loss.item():.4f}")

    avg_loss = sum(metrics["losses"]) / len(metrics["losses"]) if metrics["losses"] else 0.0
    logger.info(f"✅ REM Cycle Complete. Avg Stability Loss: {avg_loss:.4f}")
    
    return {"avg_loss": avg_loss}


def synaptic_pruning(
    model,
    pruning_ratio: float = 0.15, # Prune bottom 15% of weak connections
) -> Dict[str, int]:
    """
    Prune weak synaptic connections (LoRA weights).
    Uses adaptive quantile thresholding (prunes relative to layer strength).
    """
    logger.info("✂️  Initiating Synaptic Pruning...")
    
    pruned_count = 0
    total_params = 0
    
    # We explicitly access the PEFT model structure
    target_model = getattr(model, "model", model)

    with torch.no_grad():
        for name, param in target_model.named_parameters():
            if "lora" in name and param.requires_grad:
                total_params += param.numel()
                
                # 1. Calculate the strength of this layer
                # We prune based on the Magnitude of weights
                weights_abs = param.abs()
                
                # 2. Find the threshold for the bottom X%
                # (We move to CPU for quantile calculation to save VRAM)
                threshold = torch.quantile(weights_abs.cpu(), pruning_ratio).to(param.device)
                
                # 3. Create Mask (Keep weights ABOVE threshold)
                mask = weights_abs > threshold
                
                # 4. Count Pruned
                pruned_in_layer = (~mask).sum().item()
                pruned_count += pruned_in_layer
                
                # 5. Apply Pruning (Zero out weak weights)
                param.data *= mask.float()
                
                # 6. CRITICAL FIX: Zero out AdamW optimizer states
                # Without this, momentum (exp_avg) and variance (exp_avg_sq)
                # will resurrect pruned weights on the next training step.
                optimizer = getattr(model, "optimizer", None)
                if optimizer and param in optimizer.state:
                    state = optimizer.state[param]
                    if "exp_avg" in state:
                        state["exp_avg"] *= mask.float()
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"] *= mask.float()
                
    pruning_pct = (pruned_count / total_params * 100) if total_params > 0 else 0
    logger.info(f"✅ Pruned {pruned_count} synapses ({pruning_pct:.2f}% volume reduction)")
    
    return {
        "pruned_count": pruned_count,
        "pruning_ratio": pruning_ratio
    }

def full_sleep_cycle(model, active_topics: List[str] = None) -> Dict:
    """
    Execute a complete sleep cycle: Dream (Strengthen) -> Prune (Clean).
    """
    
    logger.info(f"🌙 Entering Deep Sleep Cycle #{getattr(model, 'sleep_cycles', 0) + 1}...")    
    # Default topics if none provided
    if not active_topics:
        active_topics = ["The nature of AI is", "Rui is", "System optimization involves"]
        
    # Phase 1: Dreaming (Consolidation)
    dream_stats = dream_cycle(model, active_topics)
    
    # Phase 2: Pruning (Cleanup)
    prune_stats = synaptic_pruning(model)
    
    # Phase 3: Consolidation (STM -> LTM Transfer)
    # We merge the surviving, strong weights into Long Term storage
    merge_stm_to_ltm(model, merge_ratio=0.1)
    
    # Phase 4: Reset STM and reload consolidated LTM
    # Without this, the adapter accumulates all gradients forever
    # and never actually "forgets" fast-learned noise.
    reset_stm(model)
    load_ltm_to_stm(model)
    
    # Update Cycle Counter
    if hasattr(model, "sleep_cycles"):
        model.sleep_cycles += 1
        
    logger.info("☀️  Waking up refreshed.")
    
    return {**dream_stats, **prune_stats}