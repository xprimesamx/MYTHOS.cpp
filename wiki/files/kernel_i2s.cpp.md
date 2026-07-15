# `kernel_i2s.cpp` — I2S Kernel

**Path:** `src/kernel_i2s.cpp`

Integer-to-Short (I2S) conversion kernel for efficient data type conversion.

## Function

```cpp
void i2s_convert(const int* input, short* output, int n);
```

Converts 32-bit integer tensor data to 16-bit short for memory-efficient storage during quantized operations.

## Implementation

Simple vectorized loop with bounds checking:
```cpp
for (int i = 0; i < n; i++) {
    output[i] = (short)std::clamp(input[i], -32768, 32767);
}
```

## Use Cases

- OIL8 centroid index packing
- Quantized weight conversion
- Memory buffer compression
