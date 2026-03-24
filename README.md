# Project: Synaptic Consolidation LLM ("LivingLLM")


**Goal:** Create a continuous learning LLM that mimics biological memory consolidation (Wake/Sleep cycles) to solve Catastrophic Forgetting.

## 1. Core Architecture

The system splits the LLM into three biological components to balance **Plasticity** (Learning speed) and **Stability** (Memory retention).

- **The Cortex (Base Model):** `Llama-3.2-3B-Instruct` loaded in 4-bit (NF4). Frozen parameters. Represents long-term stable knowledge.
- **The Hippocampus (STM Adapter):** A high-rank (`r=32`) LoRA adapter. It is active during the "Wake" state and learns rapidly via backpropagation on single inputs.
- **The Sleep Cycle (Consolidation):** A biological maintenance process that runs periodically:
  - **Dreaming (Generative Replay):** The model hallucinates continuations of recent topics to reinforce neural pathways.
  - **Pruning (Synaptic Scaling):** Weak LoRA weights (noise) are zeroed out via adaptive quantile thresholding.
  - **Consolidation (LTM Transfer):** Surviving weights are merged into a secondary "Long Term Memory" state on disk using Exponential Moving Average (EMA).

## 2. File Structure

The project is modularized into `src/` to separate logic from execution.

**Plaintext**

```
synaptic-llm/
├── requirements.txt         # Dependencies (bitsandbytes, peft, torch)
├── run_simulation.py        # Main CLI (Chat, !learn, !sleep)
└── src/
    ├── model.py             # LivingLLM Class (Model loading, Wake logic)
    ├── sleep.py             # Sleep Cycle (DreamGenerator, Pruning)
    ├── memory.py            # LTM Persistence (Disk saving, EMA merge)
    └── utils.py             # Helper tools (Safety hooks, Logging)
```

## 3. The Codebase (Latest Version)

### `src/model.py`

_Handles model initialization, 4-bit loading, and the 'Wake' learning loop._

**Python**

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, PeftModel
import logging
from .sleep import full_sleep_cycle
from .memory import load_ltm_to_stm

logger = logging.getLogger(__name__)

class LivingLLM:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", device: str = "cuda"):
        self.device = device
        self.sleep_cycles = 0

        print(f"🧠 INITIALIZING CORTEX: {model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # STM Adapter (High Plasticity)
        peft_config = LoraConfig(
            r=32, lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        # Load Long-Term Memory if exists
        load_ltm_to_stm(self)
        print("✅ System Online.")

    def talk(self, prompt: str) -> str:
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

    def learn(self, text: str, steps: int = 5):
        self.model.train()
        print(f"📝 Imprinting: '{text}'")
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        for i in range(steps):
            outputs = self.model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if i==0 or (i+1)%2==0: print(f"   Step {i+1}/{steps} | Loss: {loss.item():.4f}")

    def sleep(self):
        print("\n💤 Initiating Sleep Cycle...")
        stats = full_sleep_cycle(self)
        print(f"✅ Sleep Complete. Cycle #{self.sleep_cycles}")
```

### `src/sleep.py`

_Handles Generative Replay (Dreaming) and Adaptive Pruning._

**Python**

```
import torch
import random
import logging
from typing import List, Dict
from .memory import merge_stm_to_ltm, reset_stm

logger = logging.getLogger(__name__)

class DreamGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_dream(self, seed: str) -> str:
        self.model.eval()
        inputs = self.tokenizer(seed, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.8)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def dream_cycle(model, topics: List[str], num_dreams: int = 5) -> Dict:
    logger.info(f"💤 REM Sleep ({num_dreams} cycles)...")
    model.model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.model.parameters()), lr=1e-5)
    dream_gen = DreamGenerator(model.model, model.tokenizer)
    losses = []

    for i in range(num_dreams):
        seed = random.choice(topics) if topics else "The system logic is"
        dream = dream_gen.generate_dream(seed)
        model.model.train()

        inputs = model.tokenizer(dream, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        loss = model.model(**inputs, labels=inputs.input_ids).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return {"avg_loss": sum(losses)/len(losses) if losses else 0}

def synaptic_pruning(model, ratio: float = 0.15) -> Dict:
    logger.info("✂️ Synaptic Pruning...")
    count = 0
    target = getattr(model, "model", model)
    with torch.no_grad():
        for name, param in target.named_parameters():
            if "lora" in name and param.requires_grad:
                threshold = torch.quantile(param.abs().cpu(), ratio).to(param.device)
                mask = param.abs() > threshold
                count += (~mask).sum().item()
                param.data *= mask.float()
    return {"pruned": count}

def full_sleep_cycle(model, topics=None) -> Dict:
    if not topics: topics = ["Rui is", "The core system is"]
    d_stats = dream_cycle(model, topics)
    p_stats = synaptic_pruning(model)
    merge_stm_to_ltm(model, merge_ratio=0.1)
    if hasattr(model, "sleep_cycles"): model.sleep_cycles += 1
    return {**d_stats, **p_stats}
```

### `src/memory.py`

_Handles Disk Persistence and EMA Merging._

**Python**

```
import torch
import torch.nn as nn
import logging
import os

logger = logging.getLogger(__name__)
LTM_FILE = "checkpoints/ltm_state.pt"

def merge_stm_to_ltm(model, merge_ratio: float = 0.1):
    logger.info(f"🔄 Merging STM -> LTM (Ratio={merge_ratio})")
    os.makedirs("checkpoints", exist_ok=True)
    stm_state = {k: v.clone().cpu() for k, v in model.model.named_parameters() if "lora" in k}

    if os.path.exists(LTM_FILE):
        ltm_state = torch.load(LTM_FILE)
    else:
        ltm_state = stm_state

    merged = {}
    for k in stm_state:
        if k in ltm_state:
            merged[k] = (ltm_state[k] * (1 - merge_ratio)) + (stm_state[k] * merge_ratio)
        else:
            merged[k] = stm_state[k]

    torch.save(merged, LTM_FILE)

def load_ltm_to_stm(model):
    if not os.path.exists(LTM_FILE): return
    logger.info("📂 Loading LTM...")
    ltm_state = torch.load(LTM_FILE)
    target = getattr(model, "model", model)
    target.load_state_dict({k: v.to(model.device) for k, v in ltm_state.items() if k in target.state_dict()}, strict=False)
```

### `run_simulation.py`

_The CLI Entry Point._

**Python**

```
import sys, warnings
from src.model import LivingLLM

warnings.filterwarnings("ignore")
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

def main():
    print("--- SYNAPTIC CONSOLIDATION EXPERIMENT ---")
    try:
        brain = LivingLLM(MODEL_NAME)
        print("System Online. Commands: !learn [text], !sleep, !exit")
    except Exception as e:
        print(f"CRITICAL FAIL: {e}")
        return

    while True:
        try:
            u_in = input("\nUser: ").strip()
            if not u_in: continue
            if u_in in ["!exit", "quit"]: break

            if u_in.startswith("!learn"):
                brain.learn(u_in.replace("!learn", "").strip(), steps=5)
            elif u_in.startswith("!sleep"):
                brain.sleep()
            else:
                print(f"BioLLM: {brain.talk(u_in)}")

        except KeyboardInterrupt: continue
        except Exception as e: print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

## 4. Current Status

- **Ready for Test:** The architecture is complete.
- **Optimization:** Memory managed for 16GB VRAM (4-bit loading, CPU offloading for stats).
- **Next Action:** Run `run_simulation.py`, verify that `!learn` works, and confirm `!sleep` creates a `checkpoints/ltm_state.pt` file.
