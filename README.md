# Project: Synaptic Consolidation LLM ("LivingLLM")

**Goal:** Create a continuous learning LLM that mimics biological memory consolidation (Wake/Sleep cycles) to solve Catastrophic Forgetting.

## 1. Core Architecture

The system splits the LLM into three biological components to balance **Plasticity** (Learning speed) and **Stability** (Memory retention).

### 1.1 The Cortex (Base Model)

`Qwen/Qwen3-8B` (or a Qwen 3.x MoE variant) loaded in 4-bit (NF4). Frozen parameters. Represents long-term stable knowledge.

#### Planned Improvement: Neocortical Integration

|              |                                                                                                                                                                                   |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What**     | Periodic merge-and-quantize cycle where stable LTM weights are merged into the base model via `peft`'s `merge_and_unload()`, then re-quantized.                                   |
| **Changes**  | The Cortex transitions from a static frozen backbone to a slowly-evolving knowledge store, mirroring real neocortical learning.                                                   |
| **Value**    | True CLS (Complementary Learning Systems) dynamics — the base model itself accumulates knowledge over many sleep cycles instead of all knowledge living in fragile LoRA adapters. |
| **Priority** | 🔵 P3 — High risk, high reward. Only viable after many stable sleep cycles.                                                                                                       |

---

### 1.2 The Hippocampus (STM Adapter)

A high-rank (`r=32`) LoRA adapter. Active during the "Wake" state, learns rapidly via backpropagation on single inputs.

#### Planned Improvement: Dual LoRA System (STM + LTM Adapters)

|              |                                                                                                                                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **What**     | Replace the single LoRA adapter with **two** adapters: a high-rank STM adapter (`r=32`, high LR) for fast learning, and a low-rank LTM adapter (`r=8`, low LR) for consolidated knowledge.                         |
| **Changes**  | During Wake, only STM is active. During Sleep, STM knowledge is distilled into LTM via interleaved replay, then STM is reset. This provides true structural separation between fast and slow learning.             |
| **Value**    | Achieves actual CLS dynamics instead of simulating them with checkpoints. The structural difference in rank creates a natural information bottleneck — only the most important information survives consolidation. |
| **Priority** | 🟡 P1 — Core architectural fix that unlocks the system's thesis.                                                                                                                                                   |

#### Planned Improvement: EWC Regularization

|              |                                                                                                                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What**     | Add Elastic Weight Consolidation during `learn()`. An L2 penalty weighted by Fisher Information prevents the STM adapter from overwriting important consolidated knowledge.                 |
| **Changes**  | `total_loss = task_loss + λ * Σ(F_i * (θ_i - θ*_i)²)` where `F_i` is the Fisher Information and `θ*` is the LTM anchor.                                                                     |
| **Value**    | Prevents the current failure mode of overfitting to single inputs. The regularizer acts as a "memory protection" mechanism — new learning is balanced against existing knowledge stability. |
| **Priority** | 🟢 P2 — Standard continual learning technique that directly addresses catastrophic forgetting.                                                                                              |

---

### 1.3 The Sleep Cycle (Consolidation)

A biological maintenance process that runs periodically:

- **Dreaming (Generative Replay):** The model hallucinates continuations of recent topics to reinforce neural pathways.
- **Pruning (Synaptic Scaling):** Weak LoRA weights (noise) are zeroed out via adaptive quantile thresholding.
- **Consolidation (LTM Transfer):** Surviving weights are merged into a secondary "Long Term Memory" state on disk using Exponential Moving Average (EMA).

#### Planned Improvement: Episodic Memory Buffer

|              |                                                                                                                                                                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What**     | A ring buffer that stores actual learned inputs (text + loss at time of learning) with a configurable max size. During dreaming, seeds are sampled from this buffer using prioritized experience replay instead of using hardcoded topic strings. |
| **Changes**  | Replaces the static `["Rui is", "The core system is"]` topic list. High-loss (poorly consolidated) memories are sampled more frequently, ensuring the model rehearses what it hasn't mastered yet.                                                |
| **Value**    | Grounds dreaming in real data instead of hallucinations-of-hallucinations. Eliminates the self-reinforcing feedback loop that currently causes semantic drift.                                                                                    |
| **Priority** | 🔴 P0 — Low effort, immediately fixes the biggest architectural flaw.                                                                                                                                                                             |

#### Planned Improvement: Frozen Reference + KL-Divergence Dreaming

|              |                                                                                                                                                                                                                                                    |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What**     | Before each sleep cycle, snapshot the current model as a frozen reference. Dreams are generated by the frozen copy, and the live model trains on those dreams with an additional KL-divergence loss against the frozen copy's output distribution. |
| **Changes**  | Dream training loss becomes `L = L_replay + α * KL(P_frozen ∥ P_live)`. The frozen reference anchors the model's output distribution and prevents unconstrained drift during replay.                                                               |
| **Value**    | Transforms dreaming from "train on your own noise" to actual pseudo-rehearsal with a consistency constraint. This is how modern continual learning methods (e.g., LwF, DGR) do it properly.                                                        |
| **Priority** | 🟡 P1 — Medium effort, makes the replay mechanism actually useful.                                                                                                                                                                                 |

#### Planned Improvement: MoE Router-Guided Dynamic Rank

|              |                                                                                                                                                                                                                                                    |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What**     | Migrate the Cortex to a Mixture of Experts (MoE) architecture (e.g., Qwen 3.x MoE). Introduce a custom LoRA module where the active rank ($r$) is dynamically controlled via a gating mask tied directly to the MoE Router's activation confidence. |
| **Changes**  | `Output = Input + (B × Mask × A)`. For simple facts, the Router allocates $r=4$. For complex reasoning, it opens to $r=32$. During the Sleep phase, pruning evaluates historical mask usage to selectively delete unused ranks locally per expert. |
| **Value**    | Highly biological. Prevents simple declarative facts ("Rui is emperor") from overwriting complex reasoning pathways by restricting the cognitive bandwidth used for minor updates. Makes the pruning phase vastly more intelligent.                 |
| **Priority** | 🟣 Experimental — High effort, requires a custom PEFT module. Represents the ultimate evolution of the architecture.                                                                                                                              |

#### Planned Improvement: Router-Biased Dreaming

|              |                                                                                                                                                                                                                                                    |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What**     | Hook into the MoE Router telemetry during the Wake phase to log which specific experts were active. During the Sleep phase, generate dreams that artificially force routing through those recently activated experts.                              |
| **Changes**  | Replaces global randomized dreaming with localized rehearsal targeted strictly at the neural clusters that learned new information that day.                                                                                                       |
| **Value**    | Massively reduces VRAM and compute during sleep, and ensures dreams are relevant to the day's waking experiences rather than generic hallucinations.                                                                                               |
| **Priority** | 🟣 Experimental — Highly synergistic with the Dynamic Rank improvement.                                                                                                                                                                            |

#### Planned Improvement: Fisher Information Pruning

|              |                                                                                                                                                                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What**     | Replace blind magnitude thresholding with importance-aware pruning. Squared gradients are accumulated during `learn()` as a Fisher Information estimate. Pruning decisions use `importance = |weight| × fisher_score`. |
| **Changes**  | Weights that are small but sit on critical computation paths are preserved. Only weights that are both small _and_ unimportant to the loss landscape are pruned.                                                          |
| **Value**    | Prevents "accidental lobotomy" — the current pruner can't distinguish between a small-but-critical weight and genuine noise. Fisher-weighted pruning is the standard approach in neural network compression for a reason. |
| **Priority** | 🟢 P2 — Medium effort, directly improves pruning quality.                                                                                                                                                                 |

#### Planned Improvement: Adaptive Consolidation Rate

|              |                                                                                                                                                                                                    |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What**     | Replace the fixed `merge_ratio=0.1` with a per-parameter adaptive rate based on weight stability across sleep cycles. `stability = 1 / (1 + variance)`, then `ratio_i = base_ratio × stability_i`. |
| **Changes**  | Stable weights (low variance across cycles) consolidate faster into LTM. Volatile weights (still being actively modified) consolidate slower, giving them more time to settle.                     |
| **Value**    | Mirrors biological consolidation — well-rehearsed memories transfer to long-term storage faster than novel, still-forming memories. Prevents premature consolidation of unstable representations.  |
| **Priority** | 🔵 P3 — Low effort, nice-to-have optimization.                                                                                                                                                     |

---

### 1.4 Evaluation Framework

Allows tracking catastrophic forgetting, plasticity limits, and stability over time.

#### Planned Improvement: Forgetting Metrics & Probe Sets

|              |                                                                                                                                                                                                                                                 |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What**     | A held-out probe set of question-answer pairs from all previously learned material. After each sleep cycle, the model is evaluated against all probes. Metrics tracked: per-task accuracy, backward transfer (BWT), and forward transfer (FWT). |
| **Changes**  | Uses the `utils.py` evaluation functions. Each `learn()` call registers probe pairs. Each `sleep()` cycle runs evaluation and updates metrics.                                                                                                  |
| **Value**    | **Cannot claim to solve catastrophic forgetting without measuring forgetting.** This provides the validation suite required to quantify learning and retention performance.                                                                     |
| **Priority** | 🔴 P0 — Non-negotiable. Without this, the project is a demo, not an experiment.                                                                                                                                                                 |

---

## 2. File Structure

The project has been modularized and expanded to separate core execution logic, logging, experiments, and visualizations:

```plaintext
synaptic-consolidation-llm/
├── LICENSE
├── README.md
├── RECOMMENDATIONS.md       # Design review suggestions & code improvement notes
├── requirements.txt         # Project package dependencies
├── run_simulation.py        # Simulation test script using GPT-2 model (Draft)
├── docs/
│   ├── README.md            # Supporting documentation overview
│   └── architecture.md      # Dual-Memory Architecture Mermaid diagrams
├── experiments/
│   ├── generate_mock_plots.py # Plot generator script for presenting simulated performance
│   ├── logs/                # Simulation run logs (json metrics)
│   └── plots/               # Visual graphs showing training metrics (loss, retention, etc.)
├── notebooks/
│   ├── 01_proof_of_plasticity.ipynb
│   ├── 02_proof_of_forgetting.ipynb
│   └── 03_sleep_stabilization.ipynb
└── src/
    ├── __init__.py          # Module index exports
    ├── model.py             # LivingLLM Class (Model loading, Wake/Imprint phase)
    ├── sleep.py             # Sleep Cycle (Dreaming/Generative Replay, Synaptic Pruning)
    ├── memory.py            # Long-Term Memory (Disk saving, EMA merge, STM Reset)
    ├── utils.py             # Helper tools (Hooks, plasticity monitoring, forgetting calculation)
    └── visualization.py     # Training and consolidation metrics graphing utility
```

---

## 4. Improvement Roadmap

| Priority | Improvement                    | Effort | Impact                            |
| -------- | ------------------------------ | ------ | --------------------------------- |
| 🔴 P0    | Episodic Memory Buffer         | Low    | Fixes hallucination feedback loop |
| 🔴 P0    | Evaluation Framework & Probes  | Low    | Enables scientific validity       |
| 🟡 P1    | Frozen Reference + KL Dreaming | Medium | Makes replay mechanism useful     |
| 🟡 P1    | Dual LoRA Adapters (STM/LTM)   | Medium | Core architectural fix            |
| 🟢 P2    | EWC Regularization             | Medium | Proper continual learning         |
| 🟢 P2    | Fisher Information Pruning     | Medium | Smarter, safer pruning            |
| 🔵 P3    | Adaptive Consolidation Rate    | Low    | Biological fidelity               |
| 🔵 P3    | Neocortical Integration        | High   | True CLS dynamics                 |
| 🟣 Exp   | MoE Router-Guided Dynamic Rank | High   | Smart plasticity & memory scoping |
| 🟣 Exp   | Router-Biased Dreaming         | High   | Targeted sleep consolidation      |

---

## 5. Current Status

- **Drafting & Modularization Complete:** The project has been restructured with utility modules (`src/utils.py`), plotting mechanisms (`src/visualization.py`), and a dedicated `experiments/` directory.
- **Simulation Runner Drafted:** A simulation runner script `run_simulation.py` has been drafted to automate wake/sleep sequences using smaller architectures (`gpt2`), though it currently contains placeholder logic and API parameter mismatches under active development.
- **VRAM Optimizations:** Configured for 16GB VRAM GPUs (using 4-bit NF4 double-quantization loading parameters, moving quantile evaluations to CPU, and clearing optimizer momentum arrays during synaptic pruning to prevent memory resurrecting).
