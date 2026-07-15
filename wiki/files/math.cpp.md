# `math.cpp` — Math Operations (C++20 Fallback)

**Path:** `src/math.cpp`

Pure C++20 fallback implementations for all math operations. Used when AVX2 is not available.

## Implemented Operations

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `matmul` | O(N³) | Simple triple-loop matmul |
| `rms_norm` | O(N) | Numerically stable |
| `softmax` | O(N²) | Online softmax algorithm |
| `silu` | O(N) | `x * sigmoid(x)` |
| `gelu` | O(N) | Gaussian error linear unit |
| `relu` | O(N) | `max(0, x)` |
| `tanh` | O(N) | Hyperbolic tangent |
| `rope` | O(N) | Rotary position embeddings |
| `add` / `mul` | O(N) | Element-wise operations |
| `scalar_mul` | O(N) | Broadcast scalar multiply |
| `mean` / `var` | O(N) | Statistical moments |
| `gather` | O(N) | Index-based selection |
| `scatter_add` | O(N) | Sparse gradient accumulation |

All operations include autograd-compatible implementations that register gradient functions.
