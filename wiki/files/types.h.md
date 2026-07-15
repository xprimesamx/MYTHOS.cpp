# `types.h` — Core Type System

**Path:** `include/oil/types.h`

Defines the foundational data types and format enums used throughout MYTHOS.cpp. This is the base header that几乎所有其他文件都包含。

## Format Enum

```cpp
enum class Format : uint8_t {
    BINARY  = 0, // 1-bit, {-1, +1}
    TERNARY = 1, // 1.58-bit, {-1, 0, +1}
    OIL4    = 2, // 4-bit, codebook 16 × FP16
    OIL8    = 3, // 8-bit, codebook 256 × FP32
    FP16    = 4, // 16-bit, native half
    FP32    = 5, // 32-bit, native float
};
```

Representation formats for model weights. Each format has a corresponding bits-per-weight (BPW).

### Helper Functions

| Function | Description |
|----------|-------------|
| `format_name(Format)` | Returns human-readable name (`"binary"`, `"oil4"`, etc.) |
| `format_bpw(Format)` | Returns bits-per-weight (1.0, 1.58, 4.0, 8.0, 16.0, 32.0) |

## DType Enum

```cpp
enum class DType : uint8_t {
    I64, // int64_t
    U8,  // uint8_t
    U4,  // 4-bit packed (2 per byte)
    I2,  // 2-bit ternary packed (4 per byte)
    I1,  // 1-bit binary packed (8 per byte)
    F16, // half precision
    F32, // single precision
};
```

Tensor data types. Packed types (U4, I2, I1) store multiple values per byte for memory efficiency.

### Helper Functions

| Function | Description |
|----------|-------------|
| `dtype_size(DType)` | Returns byte size per element (1 for packed types, 2 for F16, 4 for F32, 8 for I64) |
| `format_to_dtype(Format)` | Maps Format to equivalent DType |

## Shape Struct

```cpp
struct Shape {
    int64_t dims[4];
    int rank;
    int64_t numel() const;
};
```

Fixed-size 4D shape with runtime rank. Supports up to 4 dimensions (common for transformer tensors: batch, heads, seq_len, dim).

## Autograd Types

| Type | Description |
|------|-------------|
| `GradientFn` | `std::function<void()>` — backward hook |
| `TensorPtr` | `std::shared_ptr<Tensor>` — reference-counted tensor handle |

## TransformerConfig

```cpp
struct TransformerConfig {
    int64_t vocab_size = 32000;
    int64_t hidden_size = 768;
    int64_t num_layers = 12;
    int64_t num_heads = 12;
    int64_t intermediate_size = 3072;
    int64_t max_seq_len = 2048;
    float norm_eps = 1e-5f;
    Format weight_format = Format::FP16;
    bool use_moe = false;
    int64_t num_experts = 0;
    int64_t top_k = 0;
};
```

Complete model architecture configuration with defaults suitable for small to medium transformers.
