# `int8_quant.cpp` — INT8 Quantization

**Path:** `src/int8_quant.cpp`

INT8 quantization utilities for model weights and activations.

## Functions

| Function | Description |
|----------|-------------|
| `quantize_int8(tensor)` | FP32 → INT8 symmetric quantization |
| `dequantize_int8(tensor)` | INT8 → FP32 |
| `quantize_per_channel(tensor, dim)` | Per-channel INT8 quantization |

## Algorithm

```
scale = max(abs(weights)) / 127.0
quantized = round(clamp(weights / scale, -127, 127))
```

## Per-Channel Quantization

Each output channel gets its own scale for better accuracy:
```
scale[i] = max(abs(weights[i])) / 127.0
quantized[i] = round(clamp(weights[i] / scale[i], -127, 127))
```
