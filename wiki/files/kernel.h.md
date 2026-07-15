# `kernel.h` — Compute Kernels

**Path:** `include/oil/kernel.h`

Low-level compute kernels for quantization, dequantization, and matrix operations.

## Kernel Functions

### Quantization
| Function | Description |
|----------|-------------|
| `quantize_oil4(const Tensor& src, Tensor& dst)` | Quantize FP32 → OIL4 (4-bit) |
| `quantize_oil8(const Tensor& src, Tensor& dst)` | Quantize FP32 → OIL8 (8-bit centroid) |
| `quantize_binary(const Tensor& src, Tensor& dst)` | Binarize FP32 → 1-bit |
| `quantize_ternary(const Tensor& src, Tensor& dst)` | Ternarize FP32 → 1.58-bit |

### Dequantization
| Function | Description |
|----------|-------------|
| `dequantize_oil4(const Tensor& src, Tensor& dst)` | Dequantize OIL4 → FP32 |
| `dequantize_oil8(const Tensor& src, Tensor& dst)` | Dequantize OIL8 → FP32 |
| `dequantize_binary(const Tensor& src, Tensor& dst)` | Debinarize 1-bit → FP32 |
| `dequantize_ternary(const Tensor& src, Tensor& dst)` | Determine 1.58-bit → FP32 |

### Specialized Operations
| Function | Description |
|----------|-------------|
| `quantized_matmul_oil4` | Matmul with OIL4 quantized weights |
| `quantized_matmul_oil8` | Matmul with OIL8 quantized weights |
| `i2s_convert` | Integer-to-short conversion |

### Files
| Variant | File | Description |
|---------|------|-------------|
| OIL4 kernels | `kernel_oil4.cpp` | 4-bit quantization matmul |
| OIL8 kernels | `kernel_oil8.cpp` | 8-bit centroid quantization matmul |
| I2S kernel | `kernel_i2s.cpp` | Integer-to-short conversion |
| Tile kernels | `kernel_tl.cpp` | Tiled computation |
