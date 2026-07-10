#include "oil/ste_quantizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace oil {

STEQuantizer::STEQuantizer(Format target_format) : target_format_(target_format) {}

void STEQuantizer::set_target_format(Format fmt) {
    target_format_ = fmt;
}

Format STEQuantizer::target_format() const {
    return target_format_;
}

Tensor STEQuantizer::forward(const Tensor& fp32_weight) {
    // STE: forward returns a copy (identity in forward pass)
    return fp32_weight.clone();
}

float STEQuantizer::find_scale(const float* data, int64_t n) {
    float max_abs = 0;
    for (int64_t i = 0; i < n; i++) {
        float a = std::fabs(data[i]);
        if (a > max_abs) max_abs = a;
    }
    return max_abs > 1e-10f ? max_abs : 1.0f;
}

Tensor STEQuantizer::quantize_with_codebook(const Tensor& fp32_weight, CodebookOIL8& codebook) {
    int64_t n = fp32_weight.numel();
    const float* src = (const float*)fp32_weight.data();

    // Train codebook on the data if centroids are all zero
    bool needs_train = true;
    for (int i = 0; i < CodebookOIL8::SIZE; i++) {
        if (codebook.centroids[i] != 0.0f) { needs_train = false; break; }
    }
    if (needs_train) {
        codebook.train(src, (size_t)n);
    }

    Tensor quantized(Shape{n}, DType::U8);
    uint8_t* dst = (uint8_t*)quantized.data();

    for (int64_t i = 0; i < n; i++) {
        dst[i] = codebook.quantize(src[i]);
    }

    // Dequantize back to fp32 for STE (differentiable approximation)
    Tensor result(Shape{n}, DType::F32);
    float* rd = (float*)result.data();
    for (int64_t i = 0; i < n; i++) {
        rd[i] = codebook.dequantize(dst[i]);
    }

    return result;
}

Tensor STEQuantizer::quantize_with_codebook(const Tensor& fp32_weight, CodebookOIL4& codebook) {
    int64_t n = fp32_weight.numel();
    const float* src = (const float*)fp32_weight.data();

    bool needs_train = true;
    for (int i = 0; i < CodebookOIL4::SIZE; i++) {
        if (codebook.centroids[i] != 0) { needs_train = false; break; }
    }
    if (needs_train) {
        codebook.train(src, (size_t)n);
    }

    int packed_bytes = (int)((n + 1) / 2);
    Tensor quantized(Shape{packed_bytes}, DType::U4);
    uint8_t* dst = (uint8_t*)quantized.data();
    std::memset(dst, 0, (size_t)packed_bytes);

    for (int64_t i = 0; i < n; i++) {
        uint8_t idx = codebook.quantize(src[i]);
        if (i % 2 == 0) {
            dst[i / 2] = (dst[i / 2] & 0xF0) | (idx & 0x0F);
        } else {
            dst[i / 2] = (dst[i / 2] & 0x0F) | ((idx & 0x0F) << 4);
        }
    }

    Tensor result(Shape{n}, DType::F32);
    float* rd = (float*)result.data();
    for (int64_t i = 0; i < n; i++) {
        uint8_t idx;
        if (i % 2 == 0) idx = dst[i / 2] & 0x0F;
        else idx = (dst[i / 2] >> 4) & 0x0F;
        rd[i] = codebook.dequantize(idx);
    }

    return result;
}

void STEQuantizer::quantize_ternary(const float* src, uint8_t* dst, float* scale, int64_t n) {
    float s = find_scale(src, n);
    float threshold = s * 0.5f;
    *scale = s;

    int packed_size = (int)((n + 3) / 4);
    std::memset(dst, 0, (size_t)packed_size);

    for (int64_t i = 0; i < n; i++) {
        uint8_t val;
        if (src[i] > threshold) {
            val = 1;
        } else if (src[i] < -threshold) {
            val = 2; // -1 mapped to 2 in 2-bit encoding
        } else {
            val = 0;
        }
        dst[i / 4] |= (val & 0x03) << (2 * (i % 4));
    }
}

void STEQuantizer::quantize_binary(const float* src, uint8_t* dst, float* scale, int64_t n) {
    float s = find_scale(src, n);
    *scale = s;

    int packed_size = (int)((n + 7) / 8);
    std::memset(dst, 0, (size_t)packed_size);

    for (int64_t i = 0; i < n; i++) {
        if (src[i] > 0) {
            dst[i / 8] |= (1 << (i % 8));
        }
    }
}

} // namespace oil
