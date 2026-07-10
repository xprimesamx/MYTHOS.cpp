#pragma once

#include "oil/types.h"

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <limits>

namespace oil {

struct CodebookEntry {
    float val;
    float scale;
};

class CodebookOIL8 {
public:
    static constexpr int SIZE = 256;
    float centroids[SIZE];

    CodebookOIL8() {
        for (int i = 0; i < SIZE; i++) centroids[i] = 0.0f;
    }

    void train(const float* data, size_t count);
    void ema_update(const float* data, const uint8_t* assign, size_t count, float lr);
    uint8_t quantize(float val) const;
    float dequantize(uint8_t idx) const;
    size_t serialized_size() const;
    size_t serialize(uint8_t* dst) const;
    static CodebookOIL8 deserialize(const uint8_t* src, size_t& offset);
};

class CodebookOIL4 {
public:
    static constexpr int SIZE = 16;
    uint16_t centroids[SIZE];  // F16 storage

    CodebookOIL4() {
        for (int i = 0; i < SIZE; i++) centroids[i] = 0;
    }

    void train(const float* data, size_t count);
    void ema_update(const float* data, const uint8_t* assign, size_t count, float lr);
    uint8_t quantize(float val) const;
    float dequantize(uint8_t idx) const;
    size_t serialized_size() const;
    size_t serialize(uint8_t* dst) const;
    static CodebookOIL4 deserialize(const uint8_t* src, size_t& offset);

    static uint16_t float_to_half(float f);
    static float half_to_float(uint16_t h);
};

struct TernaryScale {
    float scale;
};

struct BinaryScale {
    float scale;
};

} // namespace oil
