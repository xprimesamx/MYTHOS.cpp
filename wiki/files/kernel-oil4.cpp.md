# `kernel_oil4.cpp` — OIL4 Quantization Kernel

**Path:** `src/kernel_oil4.cpp`

OIL4 quantization kernels: 4-bit quantized matrix operations with codebook lookup.

## OIL4 Format

- **4 bits per weight** (16 centroids)
- **Centroid table**: 16 × FP16 (32 bytes)
- **Index array**: 2 weights per byte
- **Effective BPW**: ~1.50

## Kernels

| Kernel | Description |
|--------|-------------|
| `quantize_oil4(float*, uint8_t*, float*, int n)` | FP32 → OIL4 |
| `dequantize_oil4(uint8_t*, float*, float*, int n)` | OIL4 → FP32 |
| `matmul_oil4(uint8_t*, float*, float*, float*, int M, int N, int K)` | Quantized matmul |

### Matmul Algorithm

1. Load 4-bit indices from weight matrix
2. Look up FP16 centroids from codebook
3. Dequantize on-the-fly to FP32
4. FMA with input activations
5. Accumulate to output

### Performance

- Memory: 16× compression vs FP32 weights
- Compute: Dequantize overhead offset by reduced memory bandwidth
- Best for: Memory-bound inference scenarios
