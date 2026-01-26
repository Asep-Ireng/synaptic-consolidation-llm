"""
Memory Management: STM <-> LTM Transfer & Persistence.
Handles the consolidation of short-term memories into long-term storage.
"""

import torch
import torch.nn as nn
import logging
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

LTM_STORAGE_FILE = "checkpoints/ltm_state.pt"

def merge_stm_to_ltm(model, merge_ratio: float = 0.1) -> None:
    """
    Consolidate STM adapter weights into LTM storage.
    Uses Exponential Moving Average (EMA) to slowly drift LTM towards STM.
    
    Equation: LTM_new = (LTM_old * (1 - ratio)) + (STM * ratio)
    """
    logger.info(f"🔄 Merging STM -> LTM (Ratio={merge_ratio})")
    
    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(LTM_STORAGE_FILE), exist_ok=True)
    
    # 1. Load existing LTM (or create from current STM if first run)
    stm_state = {k: v.clone().cpu() for k, v in model.model.named_parameters() if "lora" in k}
    
    if os.path.exists(LTM_STORAGE_FILE):
        ltm_state = torch.load(LTM_STORAGE_FILE)
    else:
        logger.info("   First consolidation: Initializing LTM from current state.")
        ltm_state = stm_state

    # 2. Perform the Mathematical Merge (EMA)
    merged_state = {}
    with torch.no_grad():
        for name in stm_state:
            if name in ltm_state:
                # Move to CPU for math to save VRAM
                stm_param = stm_state[name]
                ltm_param = ltm_state[name]
                
                # Apply EMA
                merged_state[name] = (ltm_param * (1 - merge_ratio)) + (stm_param * merge_ratio)
            else:
                merged_state[name] = stm_state[name]

    # 3. Save the new LTM state to disk
    torch.save(merged_state, LTM_STORAGE_FILE)
    logger.info("✅ LTM consolidated and saved to disk.")

def reset_stm(model) -> None:
    """
    Clear short-term memory (Hippocampus) to prepare for a new day.
    CRITICAL FIX: LoRA B-matrices must be init to ZERO, A-matrices to Random.
    """
    logger.info("🗑️  Resetting STM (Clearing Hippocampus)...")
    
    target_model = getattr(model, "model", model)
    
    with torch.no_grad():
        for name, param in target_model.named_parameters():
            if "lora" in name:
                if "lora_B" in name:
                    # FIX: Matrix B must be ZERO so adapter starts as Identity
                    nn.init.zeros_(param)
                elif "lora_A" in name:
                    # Matrix A can be random gaussian
                    nn.init.kaiming_uniform_(param, a=5**0.5)
                elif "lora_embedding" in name:
                    # If using embedding layers
                    nn.init.normal_(param, mean=0.0, std=0.02)
                    
    logger.info("✅ STM reset complete. Brain is tabula rasa.")

def load_ltm_to_stm(model) -> bool:
    """
    Loads the LTM state into the active model.
    Call this on startup to 'Remember' previous days.
    """
    if not os.path.exists(LTM_STORAGE_FILE):
        logger.info("   No LTM found (New Brain).")
        return False
        
    logger.info("📂 Loading LTM into Active Memory...")
    ltm_state = torch.load(LTM_STORAGE_FILE)
    
    target_model = getattr(model, "model", model)
    
    # Load weights
    current_state = target_model.state_dict()
    
    # Only update keys that match (safe loading)
    new_state = {}
    for k, v in ltm_state.items():
        if k in current_state:
            new_state[k] = v.to(model.device) # Move back to GPU
            
    target_model.load_state_dict(new_state, strict=False)
    logger.info("✅ Long-Term Memory Restored.")
    return True

def extract_memories(model, prompts: List[str]) -> Dict[str, torch.Tensor]:
    """
    Extract activation patterns for visualization.
    """
    memories = {}
    model.model.eval()
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device)
            # We need to hook the output, or just use the last hidden state
            outputs = model.model(**inputs, output_hidden_states=True)
            
            # Store last hidden state (The 'Thought Vector')
            # Shape: (1, seq_len, hidden_dim) -> We take the last token
            last_token_vector = outputs.hidden_states[-1][:, -1, :].cpu()
            memories[f"mem_{i}"] = last_token_vector
            
    return memories

def memory_stability_score(model, test_prompts: List[str]) -> float:
    """
    Calculate confidence (stability) on specific prompts.
    Returns: Average Probability (0.0 - 1.0)
    """
    total_prob = 0.0
    model.model.eval()
    
    if not test_prompts:
        return 0.0
        
    with torch.no_grad():
        for prompt in test_prompts:
            inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.model(**inputs, labels=inputs["input_ids"])
            
            # Perplexity-based confidence
            # Lower loss = Higher stability
            loss = outputs.loss
            prob = torch.exp(-loss).item()
            total_prob += prob
            
    return total_prob / len(test_prompts)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Memory Management Module Loaded.")