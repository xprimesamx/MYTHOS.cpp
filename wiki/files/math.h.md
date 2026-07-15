# `math.h` — Math Operations

**Path:** `include/oil/math.h`

Optimized mathematical operations with AVX2 intrinsics and pure C++20 fallback.

## Core Functions

| Function | Description |
|----------|-------------|
| `matmul(Tensor&, const Tensor&, const Tensor&)` | Matrix multiply: `C = A × B` |
| `rms_norm(Tensor&, const Tensor&, const Tensor&)` | RMS normalization |
| `softmax(Tensor&, const Tensor&)` | Softmax along last dimension |
| `rope(Tensor&, const Tensor&, int dim)` | Rotary Position Embedding |
| `silu(Tensor&, const Tensor&)` | SiLU activation |
| `add(Tensor&, const Tensor&, const Tensor&)` | Element-wise add |
| `mul(Tensor&, const Tensor&, const Tensor&)` | Element-wise multiply |
| `scalar_mul(Tensor&, float, const Tensor&)` | Scalar multiply |
| `gelu(Tensor&, const Tensor&)` | GELU activation |
| `tanh(Tensor&, const Tensor&)` | Tanh activation |
| `relu(Tensor&, const Tensor&)` | ReLU activation |
| `copy(Tensor&, const Tensor&)` | Copy data |
| `view_as(const Tensor&, const Tensor&)` | View tensor with target shape |
| `transpose(Tensor&, const Tensor&, int, int)` | Transpose two dims |
| `slice(Tensor&, const Tensor&, int, int64_t, int64_t)` | Slice tensor |
| `cat(Tensor&, const std::vector<Tensor>&, int)` | Concatenate tensors along dim |
| `gather(Tensor&, const Tensor&, const Tensor&)` | Gather elements by indices |
| `scatter_add(Tensor&, const Tensor&, const Tensor&, const Tensor&)` | Scatter add for MoE |
| `mean(Tensor&, const Tensor&)` | Compute mean |
| `var(Tensor&, const Tensor&)` | Compute variance |
| `std(Tensor&, const Tensor&)` | Compute standard deviation |
| `max(Tensor&, const Tensor&)` | Compute max |
| `sum(Tensor&, const Tensor&)` | Compute sum |

## Implementation Variants

| Variant | File | Notes |
|---------|------|-------|
| AVX2 | `math_avx2.cpp` | Hand-tuned SIMD intrinsics for x86_64 |
| Fallback | `math.cpp` | Pure C++20, no arch-specific code |

## Architecture Notes

- Matrix multiplication uses tiled approach for cache efficiency
- RMS norm is numerically stable with epsilon protection
- RoPE is applied in-place for memory efficiency
- All operations support autograd differentiation tracking
