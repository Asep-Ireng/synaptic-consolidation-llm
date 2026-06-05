# LivingLLM Architecture

This document tracks the system architecture of the Synaptic Consolidation LLM, highlighting both the conceptual flow and the practical implementation details based on the latest codebase audit.

---

## 1. Dual-Memory Concept

The architecture splits learning into a high-plasticity **Hippocampus** (STM Adapter) and a stable **Cortex** (Base Model + LTM checkpoint), coordinated via wake/sleep cycles.

```mermaid
graph TD
    classDef cortex fill:#f9d0c4,stroke:#333,stroke-width:2px;
    classDef hippo fill:#d4e6f1,stroke:#333,stroke-width:2px;
    classDef buffer fill:#e8daef,stroke:#333,stroke-width:2px;

    subgraph "Wake Phase (Active Continual Learning)"
        Prompt[User Prompt] --> Tokenizer
        Tokenizer --> Base[Frozen LLM Base<br/>'Cortex']
        Base --> STM[STM LoRA Adapter<br/>'Hippocampus']
        STM --> Output[Generation]
        
        NewData[New Datastream] -.-> |High Learning Rate| STM
    end

    subgraph "Sleep Phase (Consolidation & Pruning)"
        ReplayBuffer[(Dream Replay Buffer)] -.-> |Simulated Replay| SleepSTM[STM Adapter]
        SleepSTM -.-> |Synaptic Pruning| PrunedSTM[Pruned STM]
        PrunedSTM ==> |Merge Weights| LTM[LTM LoRA Adapter<br/>'Consolidated Cortex']
    end
    
    LTM -.-> |Next Wake Phase| Base
    
    class Base,LTM cortex;
    class STM,SleepSTM,PrunedSTM hippo;
    class ReplayBuffer buffer;
```

---

## 2. Actual Sleep Phase Implementation Mechanics

The sleep cycle is implemented as a sequential pipeline in `src/sleep.py` and `src/memory.py`. To prevent memory leaks, catastrophic forgetting, and hallucination loops, the pipeline incorporates self-critique and sensitivity-aware pruning:

```mermaid
sequenceDiagram
    autonumber
    participant ActiveModel as Active Model (LoRA)
    participant Critic as Frozen Critic Model
    participant SleepOpt as Sleep Optimizer
    participant WakeOpt as Wake Optimizer
    participant LTM as LTM Storage
    
    Note over ActiveModel, SleepOpt: PHASE 1: DREAM CYCLE (REM SLEEP)
    ActiveModel->>ActiveModel: Generate candidate dreams
    ActiveModel->>Critic: Verify logic, syntax, factual consistency
    Critic-->>ActiveModel: Return Dynamic Reward/Penalty
    ActiveModel->>SleepOpt: Policy Gradient Update (RLAIF)
    
    Note over ActiveModel, WakeOpt: PHASE 2: SYNAPTIC PRUNING
    ActiveModel->>ActiveModel: Measure loss sensitivity (Fisher Information)
    ActiveModel->>ActiveModel: Zero out weights with flat gradient impact
    ActiveModel->>WakeOpt: Mask & zero out corresponding optimizer states (exp_avg, exp_avg_sq)
    
    Note over ActiveModel, LTM: PHASE 3: EMA CONSOLIDATION
    ActiveModel->>LTM: Merge strong weights using Exponential Moving Average
    Note over LTM: LTM = LTM*(1-ratio) + STM*ratio
    
    Note over ActiveModel, WakeOpt: PHASE 4: HIPPOCAMPAL RESET
    ActiveModel->>ActiveModel: Re-initialize STM adapter weights (B matrix -> 0, A matrix -> Kaiming)
    ActiveModel->>WakeOpt: Reset active parameters' momentum histories (clear state dict)
    
    Note over LTM, ActiveModel: PHASE 5: CORTICAL RELOAD & WARM-UP
    LTM->>ActiveModel: Load consolidated LTM weights back into base
    ActiveModel->>ActiveModel: Forward pass validation batch to capture shifted base states
    ActiveModel->>ActiveModel: Calibrate new STM adapter scaling to base shift
```

---

## 3. Architectural Audit Findings

Following a systematic audit of the theoretical and practical limits of the design, several fatal bottlenecks were identified and resolved:

### 3.1 The Real-Time Parallelization Bottleneck
* **The Issue:** Standard backpropagation is strictly sequential. Running a forward pass, computing loss, and running a backward pass directly inside a live chat inference loop will absolutely nuke token-per-second generation speeds. GPUs are built for massive parallel matrix math, not step-by-step sequential gradient updates mid-generation.
* **The Fix:** Live continual learning must be decoupled from the primary inference thread. Gradients must be accumulated asynchronously or processed via delayed micro-batches so inference latency isn't bottlenecked by the backward pass.

### 3.2 The Hallucination Amplification Loop (Semantic Drift)
* **The Issue:** Dreaming by simply generating text and training on it creates a closed-loop system where the model actively optimizes for its own mistakes. Minor bad habits get permanently burned into the LTM core via the EMA merge.
* **The Fix:** Implement a Generative Verifier Loop (RLAIF) during sleep. A secondary frozen critic evaluates candidate dreams for loops, typos, and contradictions. The active STM is updated via Policy Gradient only on verified, high-quality dreams.

### 3.3 Over-Pruning Essential Structural Subspaces
* **The Issue:** A flat absolute magnitude quantile pruning rule assumes small weights ($W \approx 0.002$) are useless. In reality, these tiny weights often control vital structural triggers (like closing brackets or punctuation). Uniform pruning lobotomizes core grammar rules.
* **The Fix:** Swap magnitude pruning for **Fisher Information Pruning**. Measure the sensitivity of the loss function relative to each weight. Prune only weights with a flat gradient impact, preserving tiny weights that would otherwise cause a massive spike in loss if zeroed.

### 3.4 The Weight-Decay Alignment Shift (LTM/STM Disconnect)
* **The Issue:** Re-initializing the STM adapter (B→0, A→Kaiming) while the base model has shifted from the overnight LTM merge causes a massive Activation Distribution Shift. The STM spends the first few hours of the wake cycle outputting garbled text trying to recalibrate to the new base weights.
* **The Fix:** Introduce a brief **Warm-up Phase** upon waking. Pass a generic validation batch through the newly merged LTM core, capture hidden states, and calibrate the initialization scaling factor of the new STM adapter matrices before taking user prompts.

### 3.5 The Momentum Resurrection Vulnerability
* **The Issue:** When weights are pruned (zeroed), PyTorch optimizers like AdamW still retain momentum values (`exp_avg` and `exp_avg_sq`). In the next training step, non-zero momentum values immediately reconstruct the "pruned" weights.
* **The Fix:** Explicitly zero out the optimizer state slices corresponding to the masked weights during pruning. Furthermore, reset the STM adapter's state dictionary entirely upon waking.
