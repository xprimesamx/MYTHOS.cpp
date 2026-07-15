# `engines/OIL8/quantize.cpp` — OIL8 Quantization

**Path:** `engines/OIL8/quantize.cpp`

OIL8 quantization engine — trains codebooks and quantizes model weights to 8-bit centroids.

## Quantization Pipeline

```
FP32 Model → Calibration → Codebook Training → Quantization → OIL8 Model
```

### Steps

1. **Calibration**: Run representative data through the model, collect weight statistics
2. **Codebook Training**: K-means clustering on weight data (256 centroids)
3. **Encoding**: Replace each weight with nearest centroid index
4. **Packing**: Store centroid table + 8-bit index array

## Functions

| Function | Description |
|----------|-------------|
| `quantize_model(Model*, int num_centroids)` | Full model quantization |
| `quantize_tensor(const Tensor&, const Codebook&)` | Single tensor quantization |
| `train_codebook(const Tensor&, int k)` | K-means codebook training |
| `calibrate(Model*, DataLoader&)` | Model calibration pass |

## OIL8 Quantization Quality

| Configuration | BPW | Quality (PPL) |
|--------------|-----|---------------|
| 256 centroids | 0.85 | ~2.5 deg |
| 65536 centroids | 0.91 | ~1.5 deg |
| FP16 baseline | 16.0 | Reference |
