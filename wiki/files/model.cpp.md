# `model.cpp` — Model Implementation

**Path:** `src/model.cpp`

Implements model loading, saving, weight management, and parameter counting.

## Key Implementations

| Method | Description |
|--------|-------------|
| `Model::load(path)` | Open OIL file, read header, load config, load all tensors |
| `Model::save(path)` | Write OIL file: header + config + all tensors |
| `DenseModel::forward()` | Full forward pass through embeddings, all layers, norm, lm_head |
| `DenseModel::param_count()` | Sum of all weight tensor sizes |

## DenseModel Architecture

```
Input IDs → Embeddings → [TransformerBlock × N] → RMSNorm → Linear → Logits
```

Each `TransformerBlock`:
1. `RMSNorm` → `Attention` (residual)
2. `RMSNorm` → `FeedForward` (residual)
