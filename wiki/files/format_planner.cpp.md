# `format_planner.cpp` — Format Planning

**Path:** `src/format_planner.cpp`

Format planning and optimization: chooses best quantization per layer.

## Functions

| Function | Description |
|----------|-------------|
| `plan_format(model, target_bpw)` | Assign format per layer |
| `estimate_quality(format_per_layer)` | Estimate PPL degradation |
| `compute_size(format_per_layer)` | Total model size |

## Format Planning Algorithm

```
Input: target_bpw (e.g., 1.50)
Output: format assignment per layer

Strategy:
- Critical layers (first & last): higher precision (FP16 or OIL4)
- Middle layers: lower precision (OIL8)
- Attention layers: slightly higher precision than FFN
- Tune to meet target_bpw while maximizing quality
```

## Format Assignment

| Layer Type | Default Format | Rationale |
|-----------|---------------|-----------|
| Embeddings | FP16 | Very sensitive to quant error |
| First 2 layers | OIL4 | Critical for feature extraction |
| Middle layers | OIL8 | More robust to quantization |
| Last 2 layers | OIL4 | Critical for output quality |
| LM Head | FP16 | Directly affects vocabulary distribution |
