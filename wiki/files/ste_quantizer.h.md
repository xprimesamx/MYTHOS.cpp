# `ste_quantizer.h` — Straight-Through Estimator Quantizer

**Path:** `include/oil/ste_quantizer.h`

Straight-Through Estimator (STE) for differentiable quantization during training.

## STEQuantizer Class

```cpp
class STEQuantizer {
    Format target_format;
    Tensor scale;
    Tensor zero_point;
    
    STEQuantizer(Format format = Format::OIL4);
    
    Tensor quantize_forward(const Tensor& x);
    Tensor quantize_backward(const Tensor& x);
    void update_scales(const Tensor& stats);
};
```

### How STE Works

1. **Forward**: Round/quantize the weights (non-differentiable)
2. **Backward**: Pretend quantization didn't happen — pass gradients through unchanged
3. **Update**: Scales adjust based on weight statistics during training

### Benefits

- Train models that are robust to quantization
- Post-training quantization quality improves
- No need for full-precision during inference
- Works with OIL4 and OIL8 formats

### Usage

```cpp
STEQuantizer quantizer(Format::OIL4);
Tensor quantized = quantizer.quantize_forward(weights);
Tensor loss = some_loss(quantized, target);
loss.backward();  // gradients skip the quantization step
```
