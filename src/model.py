"""
LivingLLM: Dual-Memory Architecture implementation.
Optimized for RTX 5070 Ti (16GB) using 4-bit Quantization.

Components:
- Cortex: Frozen Llama-3.2-3B (Base Model)
- Hippocampus: High-Rank LoRA Adapter (Plasticity Layer)
- Sleep System: Interface to consolidation & pruning logic
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, PeftModel
import logging

# Import our custom modules
from .sleep import full_sleep_cycle

logger = logging.getLogger(__name__)

class LivingLLM:
    """
    A biologically-inspired LLM that learns continuously.
    
    Lifecycle:
    1. WAKE: High-plasticity learning (updates LoRA weights via Backprop).
    2. SLEEP: Consolidation (Generative Replay + Synaptic Pruning).
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model_name = model_name
        self.sleep_cycles = 0
        
        print(f"🧠 INITIALIZING CORTEX: {model_name}")
        
        # 1. QUANTIZATION CONFIG (Critical for 16GB VRAM)
        # We use NF4 (Normal Float 4-bit) for best accuracy-to-memory ratio
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # 2. LOAD BASE MODEL (The Frozen Cortex)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 3. CONFIGURE HIPPOCAMPUS (The Plastic Adapter)
        # We use a relatively high rank (r=32) to allow "Fast Binding" of new facts.
        # During sleep, we will prune the weak connections here.
        peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Attach the adapter
        self.model = get_peft_model(self.model, peft_config)
        
        # 4. INITIALIZE OPTIMIZER (The Neurotransmitter)
        # We keep this persistent so momentum isn't lost between 'learn' calls
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        self.model.print_trainable_parameters()
        print("✅ System Online. Hippocampus ready for imprinting.")

    def talk(self, prompt: str, max_length: int = 100) -> str:
        """
        Inference: Read-only access to current memory.
        """
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new part (response)
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text.replace(prompt, "").strip()

    def learn(self, text: str, steps: int = 5):
        """
        WAKE STATE: Rapid encoding of new information.
        This mimics 'One-Shot' or 'Few-Shot' biological learning.
        """
        self.model.train()
        print(f"📝 Imprinting Memory: '{text}'")
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        labels = inputs.input_ids.clone()
        
        # Training Loop (Overfitting on purpose to simulate "Flashbulb Memory")
        for i in range(steps):
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            
            # Backprop
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if (i+1) % 2 == 0 or i == 0:
                print(f"   Step {i+1}/{steps} | Loss: {loss.item():.4f}")

    def sleep(self):
        """
        SLEEP STATE: Trigger the consolidation process.
        """
        print("\n💤 Initiating Sleep Cycle...")
        
        # We define a few 'Core Topics' to maintain during dreams.
        # In a full version, this would be dynamic based on usage history.
        active_topics = ["The identity of Rui", "The purpose of this AI", "Recent user inputs"]
        
        # Run the cycle (Dream -> Prune)
        stats = full_sleep_cycle(self, active_topics)
        
        print(f"✅ Sleep Complete. Cycle #{self.sleep_cycles}")
        print(f"   Stability Loss: {stats.get('avg_loss', 0):.4f}")
        print(f"   Pruned Synapses: {stats.get('pruned_count', 0)}")