# `math_avx2.cpp` — AVX2 Optimized Math

**Path:** `src/math_avx2.cpp`

AVX2 SIMD-optimized implementations using Intel intrinsics. Requires CPU with AVX2 support (Haswell 2013+).

## Optimized Operations

| Operation | Speedup vs C++ | Details |
|-----------|---------------|---------|
| `matmul` | 4-8× | Tiled + vectorized, 8 floats per instruction |
| `rms_norm` | 3-5× | Vectorized norm computation |
| `softmax` | 3-4× | Vectorized exp + sum + normalize |
| `silu` | 4-6× | Vectorized sigmoid with polynomial approx |
| `add` / `mul` | 8× | Full 256-bit vector width |
| `scalar_mul` | 8× | Broadcast + vectorized |

## Techniques Used

- **256-bit YMM registers**: 8 single-precision ops per instruction
- **FMA (Fused Multiply-Add)**: `_mm256_fmadd_ps` for matmul
- **Tiling**: Cache-friendly tile sizes (64×64)
- **Prefetch**: Software prefetch for large matrices
- **Loop unrolling**: Manual unrolling for smaller ops

## Fallback

If AVX2 is not detected at build time, the pure C++20 fallback in `math.cpp` is used instead.
