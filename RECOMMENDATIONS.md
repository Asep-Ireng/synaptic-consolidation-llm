# Recommendations & Observations

> Notes from code review — not criticisms, just things to consider if you ever pick this back up.

---

## 1. Dream Replay Coverage

**Current State:** Hardcoded topic seeds in `sleep.py:172`

```python
if not topics: topics = ["Rui is", "The core system is"]
```

**The Issue:** Dreams don't replay _what was actually learned_. If you `!learn` about quantum physics, the sleep cycle still dreams about "Rui is..."

**Suggested Fix:** Track recent learning topics in `LivingLLM`:

```python
# In model.py
self.recent_topics = []

def learn(self, text: str, steps: int = 5):
    self.recent_topics.append(text[:50])  # Store first 50 chars as seed
    # ... existing learning code

def sleep(self):
    stats = full_sleep_cycle(self, topics=self.recent_topics or None)
    self.recent_topics = []  # Clear after consolidation
```

---

## 2. Aggressive Cumulative Pruning

**Current State:** 15% pruning every sleep cycle (`sleep.py:158`)

**The Issue:** After ~7 cycles, you've potentially touched most weights. Compounding pruning can destabilize learned representations.

**Suggested Fix:** Add maturity-based pruning or decay the ratio:

```python
def synaptic_pruning(model, base_ratio: float = 0.15) -> Dict:
    # Reduce aggressiveness as model matures
    effective_ratio = base_ratio * (0.9 ** model.sleep_cycles)
    # ... rest of pruning logic using effective_ratio
```

---

## 3. Unused Forgetting Metric

**Current State:** `calculate_catastrophic_forgetting()` exists in `utils.py:105-131` but is never called.

**Suggested Fix:** Add before/after measurement in sleep cycle:

```python
# In full_sleep_cycle()
baseline_ppl = measure_baseline(model, topics)
# ... do dream and prune ...
forgetting = calculate_catastrophic_forgetting(model, topics, baseline_ppl)
logger.info(f"Forgetting delta: {sum(forgetting.values()):.4f}")
```

---

## 4. Learning Rate Asymmetry

| Phase                  | Learning Rate | Notes                 |
| ---------------------- | ------------- | --------------------- |
| Wake (`model.py:80`)   | `1e-4`        | Aggressive imprinting |
| Dream (`sleep.py:141`) | `1e-5`        | Gentle reinforcement  |

This is probably intentional and makes biological sense — dreams should reinforce, not overwrite. Just documenting for future you.

---

## 5. If You Want to Make This "Real"

1. **Quantitative Experiments**
   - Measure forgetting curves with/without sleep cycles
   - Compare against naive continual learning baseline

2. **Selective Consolidation**
   - Not all memories are equal — add importance weighting
   - Emotional salience? Frequency of access?

3. **Paper Material**
   - The metaphor is solid and the implementation is clean
   - Could be a short workshop paper at a venue like TinyPapers or a student track

---

_Generated from code review — feel free to ignore if this stays a gabut project._
