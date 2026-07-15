# `codebook.h` — Vector Quantization Codebooks

**Path:** `include/oil/codebook.h`

Codebook-based vector quantization for OIL4 and OIL8 weight compression.

## Codebook Classes

```cpp
class Codebook {
    int num_centroids;
    int dim;
    Tensor centroids;  // [num_centroids, dim] FP32
    
    Codebook(int num_centroids, int dim);
    Codebook(const Tensor& centroids);
    
    Tensor quantize(const Tensor& x);
    Tensor dequantize(const Tensor& indices);
    Tensor encode(const Tensor& x);
    Tensor decode(const Tensor& encoded);
};
```

### Codebook Training

```cpp
class CodebookTrainer {
    int num_centroids;
    
    Codebook train_kmeans(const Tensor& data, int num_iterations = 100);
    Codebook train_kmeans_pp(const Tensor& data, int num_iterations = 100);
};
```

### Quantization Process

1. **Training**: K-means clustering on weight statistics
2. **Encoding**: Each weight → nearest centroid index
3. **Storage**: Store centroid table + index array
4. **Decoding**: Index → centroid value (OIL4: 16 centroids × FP16, OIL8: 256 centroids × FP32)

### Codebook Configurations

| Variant | Centroids | Codebook BPW | Effective BPW |
|---------|-----------|-------------|---------------|
| OIL4 | 16 | 16 × FP16 = 256 bits | ~1.50 |
| OIL8 (256) | 256 | 256 × FP32 = 8192 bits | ~0.85 |
| OIL8 (65536) | 65536 | 65536 × FP32 | ~0.91 |
