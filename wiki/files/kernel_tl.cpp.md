# `kernel_tl.cpp` — Tile Kernel

**Path:** `src/kernel_tl.cpp`

Tiled computation kernels for cache-friendly matrix operations.

## Functions

| Function | Description |
|----------|-------------|
| `tiled_matmul(C, A, B, tile_size)` | Tiled matrix multiply |
| `tiled_add(C, A, B, tile_size)` | Tiled element-wise add |

## Tiling Strategy

```
For each A_tile in A (tile_size × tile_size):
  For each B_tile in B:
    Load A_tile and B_tile into fast memory
    Compute partial product
    Accumulate to C_tile
```

## Tile Sizes

| Cache Level | Tile Size | Rationale |
|-------------|-----------|-----------|
| L1 (32KB) | 64×64 | Fits in L1 cache |
| L2 (256KB) | 128×128 | Fits in L2 cache |
| L3 (8MB+) | 256×256 | Fits in L3 cache |

## Benefits

- Reduces cache misses
- Better memory bandwidth utilization
- Enables autograd gradient computation in tiles
