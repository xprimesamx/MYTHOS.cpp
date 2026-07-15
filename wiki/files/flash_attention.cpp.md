# `flash_attention.cpp` — Flash Attention (CPU)

**Path:** `src/flash_attention.cpp`

Tiled attention computation for reduced memory usage — CPU implementation.

## Algorithm

```
For each query tile Q_block:
  Load Q_block → SRAM (cache)
  For each key/value tile K_block, V_block:
    Load K_block, V_block → SRAM
    Compute S = Q_block × K_blockᵀ (partial attention scores)
    Apply softmax on-the-fly (online safe softmax)
    Accumulate O += softmax(S) × V_block
  Write O_block → HBM
```

## Benefits

- No `N×N` attention matrix materialization
- Memory: O(N) instead of O(N²)
- Cache-friendly tile sizes
- Online softmax for numerical stability

## Implementation Details

```cpp
// Tile sizes tuned for L1/L2 cache
constexpr int BLOCK_SIZE = 64;  // Query tile
constexpr int KV_BLOCK = 64;    // KV tile

for (int q_start = 0; q_start < N; q_start += BLOCK_SIZE) {
    for (int kv_start = 0; kv_start < N; kv_start += KV_BLOCK) {
        // Process tile
    }
}
```
