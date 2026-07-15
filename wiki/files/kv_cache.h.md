# `kv_cache.h` — Key-Value Cache

**Path:** `include/oil/kv_cache.h`

Manages the key-value cache for efficient autoregressive generation.

## KVCache Class

```cpp
class KVCache {
    std::vector<Tensor> k_caches;  // per-layer key caches
    std::vector<Tensor> v_caches;  // per-layer value caches
    int64_t max_seq_len;
    int64_t current_seq_len;

    KVCache(int num_layers, int batch_size, int num_heads, 
            int head_dim, int max_seq_len);
    
    void append(int layer, const Tensor& k, const Tensor& v);
    std::pair<Tensor, Tensor> get(int layer) const;
    void clear();
    int64_t size() const;
    bool is_full() const;
};
```

### Operations

| Method | Description |
|--------|-------------|
| `append(layer, k, v)` | Append new K,V tensors to layer cache |
| `get(layer)` | Retrieve cached K,V for layer |
| `clear()` | Reset cache for new generation |
| `size()` | Current sequence length |
| `is_full()` | Check if cache is at capacity |

### Usage

Used during autoregressive inference to avoid recomputing attention keys and values for previous tokens. Each transformer layer maintains its own K and V cache.
