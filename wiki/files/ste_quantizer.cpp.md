# `ste_quantizer.cpp` — STE Quantizer Implementation

**Path:** `src/ste_quantizer.cpp`

Straight-Through Estimator for differentiable quantization during training.

## Forward Pass

```cpp
Tensor STEQuantizer::quantize_forward(const Tensor& x) {
    // 1. Scale input to quantization range
    Tensor scaled = x / scale;
    // 2. Round to nearest quantized value (non-diff)
    Tensor rounded = round(scaled);
    // 3. Clamp to valid range
    Tensor clamped = clamp(rounded, min_val, max_val);
    // 4. Dequantize back
    Tensor result = clamped * scale;
    return result;
}
```

## Backward Pass (STE)

```cpp
Tensor STEQuantizer::quantize_backward(const Tensor& grad_output) {
    // Straight-Through Estimator:
    // Pass gradient through as-is (identity function in backward)
    // This pretends quantization is differentiable
    return grad_output;
}
```

## Scale Updates

Scales adjust during training based on weight statistics:
```
scale = 2 * mean(abs(weights)) / sqrt(quantization_range)
```
