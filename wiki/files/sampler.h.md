# `sampler.h` — Token Sampling Strategies

**Path:** `include/oil/sampler.h`

Implements token sampling strategies for text generation.

## Sampler Class

```cpp
class Sampler {
    float temperature;
    float top_p;
    int top_k;
    int64_t seed;

    Sampler(float temperature = 1.0f, float top_p = 0.95f, 
            int top_k = 40, int64_t seed = -1);
    
    int64_t sample(const Tensor& logits);
    int64_t greedy(const Tensor& logits);
    void reset();
};
```

### Sampling Methods

| Method | Description |
|--------|-------------|
| `sample()` | Sample from logits using temperature + top-p + top-k |
| `greedy()` | Always pick the most probable token |

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `temperature` | 1.0 | Lower = more deterministic, higher = more random |
| `top_p` | 0.95 | Nucleus sampling: only tokens with cumulative probability |
| `top_k` | 40 | Only consider top K tokens |
| `seed` | random | Random seed for reproducibility |

### Algorithm

1. Apply temperature scaling: `logits /= temperature`
2. Apply top-k filtering: keep only top k logits
3. Apply top-p (nucleus) filtering: smallest set of top tokens with cumulative prob ≥ p
4. Softmax to get probabilities
5. Sample from the filtered distribution
