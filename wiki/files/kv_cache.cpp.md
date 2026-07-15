# `kv_cache.cpp` — KV Cache Implementation

**Path:** `src/kv_cache.cpp`

Manages key-value cache for efficient autoregressive generation.

## Implementation

```cpp
KVCache::KVCache(num_layers, batch, num_heads, head_dim, max_seq_len)
    // Pre-allocate all caches
    for each layer:
        k_caches[layer] = Tensor({batch, num_heads, max_seq_len, head_dim})
        v_caches[layer] = Tensor({batch, num_heads, max_seq_len, head_dim})
```

## Key Operations

| Operation | Description |
|-----------|-------------|
| `append()` | Copy new K,V into cache at current position |
| `get()` | Return slice of cache up to current length |
| `clear()` | Reset sequence length to 0 |
| Allocations | Pre-allocated at max_seq_len to avoid reallocation |

## Memory Usage

| Config | Cache Size (per layer) |
|--------|----------------------|
| 768 hidden, 12 heads, 2048 seq | 768 × 2048 × 2 = 3 MB/layer |
| 12 layers total | ~36 MB |

```
// Cache layout:
// k_caches[layer] = [batch, heads, max_seq_len, head_dim]
// v_caches[layer] = [batch, heads, max_seq_len, head_dim]
// current_seq_len tracks how much is filled
```
