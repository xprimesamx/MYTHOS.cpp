# `engine/OIL8/codec.cpp` — OIL8 Codec

**Path:** `engines/OIL8/codec.cpp`

OIL8 codec implementation — the 8-bit centroid-based quantization format.

## OIL8Codec

```
Quantization: FP32 → centroid index (8-bit)
Dequantization: centroid index → FP32 centroid value
```

### Encoding Process

1. Collect weight statistics
2. Train codebook via K-means (256 centroids)
3. Assign each weight to nearest centroid
4. Store: centroid table (256 × FP32) + index array (8-bit per weight)

### Decoding Process

1. Read centroid table
2. For each index: lookup centroid value
3. Reconstruct FP32 weight tensor

### Features

| Feature | Description |
|---------|-------------|
| Compression ratio | ~16× vs FP32, ~32× vs FP16 |
| Effective BPW | ~0.85 with 256 centroids |
| Encoding speed | ~50 MB/s |
| Decoding speed | ~500 MB/s (table lookup) |

### Files

| File | Description |
|------|-------------|
| `codec.h` | Codec interface and declarations |
| `codec.cpp` | Codec implementation |
| `quantize.h/cpp` | Quantization utilities |
