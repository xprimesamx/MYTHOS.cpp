# `sampler.cpp` — Sampler Implementation

**Path:** `src/sampler.cpp`

Implements token sampling strategies: temperature scaling, top-k filtering, top-p (nucleus) filtering.

## Sampling Algorithms

### Temperature
```cpp
logits[i] /= temperature;
// temperature = 0 → greedy (all mass on max)
// temperature < 1 → sharper distribution
// temperature > 1 → flatter distribution
```

### Top-K
Keep only the `k` highest logits, set rest to -inf.

### Top-P (Nucleus)
1. Sort logits descending
2. Compute cumulative softmax probabilities
3. Keep smallest set with cumulative prob ≥ `p`
4. Set rest to -inf

### Greedy
Always select `argmax(logits)` — deterministic.

## Functions

| Function | Description |
|----------|-------------|
| `Sampler::sample()` | Full sampling pipeline |
| `Sampler::greedy()` | Deterministic selection |
| `softmax(logits)` | Convert to probabilities |
| `filter_top_k(logits, k)` | Top-K masking |
| `filter_top_p(probs, p)` | Nucleus filtering |
