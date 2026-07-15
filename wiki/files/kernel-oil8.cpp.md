# `kernel_oil8.cpp` — OIL8 Quantization Kernel

**Path:** `src/kernel_oil8.cpp`

OIL8 quantization kernels: 8-bit centroid-based matrix operations.

## OIL8 Format

- **8 bits per weight** (256 centroids)
- **Centroid table**: 256 × FP32 (1024 bytes)
- **Index array**: 1 byte per weight
- **Effective BPW**: ~0.85

## Kernels

| Kernel | Description |
|--------|-------------|
| `quantize_oil8(float*, uint8_t*, float*, int n)` | FP32 → OIL8 |
| `dequantize_oil8(uint8_t*, float*, float*, int n)` | OIL8 → FP32 |
| `matmul_oil8(uint8_t*, float*, float*, float*, int M, int N, int K)` | Quantized matmul |
| `scatter_oil8_centroids(float*, const Tensor&, int n)` | Centroid scatter |

### Matmul Algorithm

Same as OIL4 but with 8-bit indices and FP32 centroids (higher precision at cost of more index bits).
