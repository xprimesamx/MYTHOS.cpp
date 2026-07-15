# `engines/OIL8/codec.cpp` — OIL8 Codec Implementation

**Path:** `engines/OIL8/codec.cpp`

OIL8 encoding/decoding implementation with centroid-based compression.

## Encoding

```cpp
void OIL8Codec::encode(Tensor& output, const Tensor& input, 
                        const Tensor& codebook) {
    // For each weight w in input:
    //   Find nearest centroid: argmin ||w - c|| for c in codebook
    //   Store centroid index (uint8_t)
    // Output: index array + centroid table
}
```

## Decoding

```cpp
void OIL8Codec::decode(Tensor& output, const Tensor& encoded,
                        const Tensor& codebook) {
    // For each index i in encoded:
    //   output[i] = codebook[i]
    // Pure table lookup — very fast
}
```

## Performance

| Operation | Speed |
|-----------|-------|
| Encode | ~50 MB/s |
| Decode | ~500 MB/s |
| Compression ratio | ~16× vs FP32 |
