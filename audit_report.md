# 🔬 Audit Report: synaptic-consolidation-llm

> Full codebase audit — bugs, architectural flaws, code quality, and what to do about it.

---

## Executive Summary

The concept is genuinely interesting — a bio-inspired continual learning system with Wake/Sleep cycles, LoRA-based STM/LTM, and generative replay. The README is impressively thorough and shows deep understanding of the design space.

The *code*, however, has several issues ranging from "won't run at all" to "subtly wrong in ways that corrupt your experiment." Let's go through them.

---

## 🔴 Critical Bugs (Will Crash or Produce Wrong Results)

### 1. `run_simulation.py` is broken — API mismatch with `LivingLLM`

[run_simulation.py:17](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/run_simulation.py#L17)

```python
model = LivingLLM(model_name="openai-community/gpt2", stm_rank=4, ltm_rank=8)
```

`LivingLLM.__init__()` accepts `model_name` and `device`. There's no `stm_rank` or `ltm_rank` parameter. This crashes immediately with a `TypeError`.

**Also on line 39:**
```python
train_losses.extend(sleep_metrics["dream"]["losses"])
```

`full_sleep_cycle()` returns `{**dream_stats, **prune_stats}` which flattens to `{"avg_loss": ..., "pruned_count": ..., "pruning_ratio": ...}`. There's no nested `"dream"` key. This would crash with `KeyError`.

**And line 60:**
```python
plasticity_scores=[1.0, 1 - (sleep_metrics["pruning"]["pruning_percentage"]/100)],
```

Same issue — no `"pruning"` nested key, and the field is `"pruning_ratio"`, not `"pruning_percentage"`.

> [!CAUTION]
> `run_simulation.py` is completely non-functional. It was written against a different API version than the current `src/` code. Either it was never tested, or the modules were refactored without updating the runner.

---

### 2. `torch.load()` without `weights_only=True` — Security + Deprecation Warning

[memory.py:32](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/memory.py#L32), [memory.py:89](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/memory.py#L89), [utils.py:165](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/utils.py#L165)

```python
ltm_state = torch.load(LTM_STORAGE_FILE)  # Arbitrary code execution risk
```

Since PyTorch 2.6, this throws a `FutureWarning` and will eventually error. More importantly, `torch.load` uses pickle by default, which means loading an untrusted `.pt` file can execute arbitrary code.

**Fix:** `torch.load(path, weights_only=True, map_location="cpu")`

---

### 3. Optimizer state masking references the wrong optimizer

[sleep.py:143](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/sleep.py#L143)

```python
# Inside synaptic_pruning():
optimizer = getattr(model, "optimizer", None)
if optimizer and param in optimizer.state:
```

This grabs the **wake optimizer** (`model.optimizer`), but `dream_cycle()` creates its own **separate sleep optimizer** on [line 58](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/sleep.py#L58). That sleep optimizer is local to `dream_cycle()` and is garbage-collected before pruning runs.

So what happens:
1. The sleep optimizer's momentum states are **never cleaned** (it's gone)
2. The wake optimizer's momentum states **are** cleaned, but its parameters are keyed by the **original parameter objects** before PEFT wrapping — so `param in optimizer.state` may silently return `False` depending on PEFT version

The "CRITICAL FIX" comment is correct in intent but the code doesn't actually accomplish what it claims.

---

### 4. `model.model.parameters()` in optimizer includes frozen base model params

[model.py:77](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/model.py#L77)

```python
self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
```

After `get_peft_model()`, `self.model` is a PeftModel. Calling `.parameters()` returns **all** parameters including frozen base model params. This means AdamW allocates momentum buffers for ~3B frozen parameters that never get gradients.

That's a massive VRAM waste on a system specifically optimized for 16GB.

**Fix:**
```python
self.optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4
)
```

(The sleep optimizer in `dream_cycle()` already does this correctly, ironically.)

---

## 🟡 Architectural Issues (Won't Crash, But Silently Wrong)

### 5. The dream loop trains on its own hallucinations — self-reinforcing drift

[sleep.py:66-93](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/sleep.py#L66-L93)

The dream cycle:
1. Generates text from the model
2. Trains the model on that generated text
3. Repeat

This is a positive feedback loop. The model reinforces its own biases and noise. After multiple sleep cycles, you get semantic drift — the model "remembers" things it hallucinated, not things it was taught.

> [!WARNING]
> The README already identifies this (Frozen Reference + KL-Divergence Dreaming is listed as P1) and the RECOMMENDATIONS.md mentions it. But it's worth flagging in the audit because **this fundamentally undermines the core claim** of the project. Right now, "dreaming" makes the model *worse*, not better.

---

### 6. STM reset after LTM merge destroys all useful state

[sleep.py:177-183](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/sleep.py#L177-L183)

```python
merge_stm_to_ltm(model, merge_ratio=0.1)  # Save 10% of STM into LTM
reset_stm(model)                            # ZERO OUT all LoRA weights
load_ltm_to_stm(model)                      # Load back from disk
```

The merge ratio is 0.1, meaning LTM only absorbs 10% of current STM. Then STM is completely wiped and reloaded from LTM. So after one cycle:
- 90% of what was just learned is gone
- LTM only captured 10% of it

This is *extremely* aggressive forgetting. After the first sleep cycle, a freshly learned fact has only 10% strength. After two cycles: ~19% (0.1 + 0.9×0.1). The EMA math is correct but the parameters make it very lossy.

Whether this is "biologically accurate" is debatable, but practically, you'd need many reinforcement cycles before anything sticks in LTM.

---

### 7. Hardcoded LTM path — breaks if run from different directories

[memory.py:14](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/memory.py#L14)

```python
LTM_STORAGE_FILE = "checkpoints/ltm_state.pt"
```

This is a relative path. If you `cd` somewhere else and import the module, the checkpoint goes to the wrong place. Or worse — you silently create a *new* LTM in the wrong directory while the real one sits unused.

**Fix:** Make it relative to the project root or accept it as a config parameter.

---

### 8. `talk()` response extraction is fragile

[model.py:99-100](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/model.py#L99-L100)

```python
full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
return full_text.replace(prompt, "").strip()
```

Using `str.replace(prompt, "")` to extract the response is fragile. If the model's output contains the prompt text naturally, or if tokenization/detokenization slightly modifies the prompt string (whitespace, special chars), you get garbled output or empty strings.

**Fix:** Decode only the generated tokens:
```python
input_length = inputs["input_ids"].shape[1]
return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
```

---

## 🔵 Code Quality & Project Hygiene

### 9. README contains an entire duplicate codebase that's out of sync

[README.md:130-401](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/README.md#L130-L401)

The README embeds the full source code of `model.py`, `sleep.py`, `memory.py`, and `run_simulation.py`. But these embedded versions **differ** from the actual files:

| Aspect | README version | Actual file |
|---|---|---|
| `LivingLLM.__init__` | Calls `load_ltm_to_stm(self)` at startup | Doesn't |
| `run_simulation.py` | Interactive CLI loop | Batch simulation with metrics |
| `model.py` | `trust_remote_code=True` | Present in actual file too |
| `sleep.py` | No logging in dream loop | Has emoji logging |

You now have two codebases to maintain. The README will inevitably rot (it already has).

> [!IMPORTANT]
> Either remove the code from the README (link to the files instead) or accept that readers will see stale code. Embedding full source in a README is an anti-pattern for exactly this reason.

---

### 10. Notebooks are empty shells

[01_proof_of_plasticity.ipynb](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/notebooks/01_proof_of_plasticity.ipynb) — 78 bytes
[02_proof_of_forgetting.ipynb](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/notebooks/02_proof_of_forgetting.ipynb) — 78 bytes
[03_sleep_stabilization.ipynb](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/notebooks/03_sleep_stabilization.ipynb) — 78 bytes

78 bytes each — these are likely just `{"cells": [], "metadata": {}}`. They signal intent but deliver nothing. Either populate them or remove them.

---

### 11. `generate_mock_plots.py` saves to wrong relative path

[generate_mock_plots.py:35](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/experiments/generate_mock_plots.py#L35)

```python
plt.savefig('plots/forgetting_curve.png', dpi=300)
```

The script lives in `experiments/` but saves to `plots/` relative to CWD, not relative to `experiments/`. If you run it from the project root, it creates `plots/` in the root instead of `experiments/plots/`.

Line 90:
```python
os.makedirs('plots', exist_ok=True)
```
Same issue — creates the wrong directory.

---

### 12. No `setup.py`, `pyproject.toml`, or entry point config

The project is imported as `from src.model import ...` which only works if you run from the project root. There's no installable package configuration. This isn't critical for a gabut project but it means:
- No `pip install -e .`
- No reproducible environment beyond `requirements.txt`
- `requirements.txt` has no version pins — `peft`, `transformers`, and `bitsandbytes` have breaking API changes between versions *regularly*

---

### 13. Mixed `print()` and `logging`

- [model.py](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/model.py): Uses `print()` for output
- [sleep.py](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/sleep.py): Uses `logger.info()` 
- [memory.py](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/memory.py): Uses `logger.info()`

If someone configures logging to write to a file, they'll only get half the output. Pick one.

---

### 14. Missing `__all__` exports for newer functions

[__init__.py](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/__init__.py) exports `monitor_plasticity` but doesn't export:
- `calculate_catastrophic_forgetting`
- `save_checkpoint` / `load_checkpoint`
- `load_ltm_to_stm`
- `DreamGenerator`

Minor, but inconsistent.

---

## 📋 Prioritized Fix List

| # | Severity | Issue | Effort | File |
|---|---|---|---|---|
| 1 | 🔴 | `run_simulation.py` API mismatch — crashes on launch | Low | [run_simulation.py](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/run_simulation.py) |
| 2 | 🔴 | Optimizer gets all params (including frozen) — VRAM explosion | Low | [model.py:77](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/model.py#L77) |
| 3 | 🔴 | `torch.load` without `weights_only=True` | Low | [memory.py](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/memory.py), [utils.py](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/utils.py) |
| 4 | 🟡 | Optimizer state masking references wrong optimizer | Medium | [sleep.py:143](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/sleep.py#L143) |
| 5 | 🟡 | Dream loop self-reinforcement (needs KL anchor) | High | [sleep.py:66-93](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/sleep.py#L66-L93) |
| 6 | 🟡 | `talk()` response extraction fragility | Low | [model.py:99](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/model.py#L99) |
| 7 | 🔵 | Hardcoded relative LTM path | Low | [memory.py:14](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/memory.py#L14) |
| 8 | 🔵 | README code drift | Low | [README.md](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/README.md) |
| 9 | 🔵 | Empty notebooks | Low | `notebooks/` |
| 10 | 🔵 | No version pins in requirements | Low | [requirements.txt](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/requirements.txt) |
| 11 | 🔵 | Mixed print/logging | Low | [model.py](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/src/model.py) |
| 12 | 🔵 | Mock plots wrong save path | Low | [generate_mock_plots.py](file:///r:/Kuliah/Kuliah%20Rui/Project%20Gabut/synaptic-consolidation-llm/experiments/generate_mock_plots.py) |

---

## 💡 What's Actually Good

Because it's not all doom and gloom:

- **The biological metaphor is well-executed.** STM/LTM, sleep consolidation, synaptic pruning — the mapping from neuroscience to ML is genuinely thoughtful, not just aesthetic naming.
- **VRAM awareness is consistent.** CPU offloading for quantile calculations, scalar-only activation monitoring, 4-bit quantization — shows real awareness of hardware constraints.
- **LoRA B→zero, A→kaiming reset** in `reset_stm()` is correct. A lot of people would just zero everything and break the initialization invariant.
- **The README roadmap** (Episodic Memory Buffer, EWC, Fisher Pruning, KL Dreaming) shows you know exactly what's missing and why. That's more important than having built it.
- **Code is readable.** Good docstrings, clear variable names, logical module separation.

---

> Want me to fix any of these? I can start with the crashes (items 1-3) and work down.
