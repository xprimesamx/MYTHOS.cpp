# `inference_opt.cpp` — Optimized Inference Implementation

**Path:** `src/inference_opt.cpp`

Optimized forward pass configurations for production inference.

## Functions

| Function | Description |
|----------|-------------|
| `prefill()` | Process prompt in parallel |
| `decode_step()` | Single token generation |

## Prefill vs Decode

```
Prefill phase:
  Input: token0, token1, ..., tokenN (all prompt tokens)
  Process: N tokens in parallel
  Output: logits for next token

Decode phase (repeated):
  Input: token_{N+1} (single token)
  Process: 1 token with KV cache
  Output: logits for next token
```

## Configuration

```cpp
InferenceConfig{
    .use_flash_attn = true,    // Memory-efficient attention
    .use_kv_cache = true,      // Cache for generation
    .quantize_matmul = true,   // Use quantized weights
    .num_threads = 4,          // Parallel threads
};
```
