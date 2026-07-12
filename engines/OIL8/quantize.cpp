#include "quantize.h"
#include "oil/tensor.h"
#include "oil/types.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cstdint>

namespace oil {
namespace oil8 {

QuantizeParams quantize_tensor(const Tensor& src, Tensor& dst) {
    OIL_CHECK(src.dtype() == DType::F32, "quantize_tensor: src must be F32");
    OIL_CHECK(src.numel() == dst.numel(), "quantize_tensor: size mismatch");

    const float* src_data = src.data<float>();
    int64_t n = src.numel();

    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float v = std::abs(src_data[i]);
        if (v > max_abs) max_abs = v;
    }
    if (max_abs == 0.0f) max_abs = 1.0f;

    QuantizeParams params;
    params.scale = 127.0f / max_abs;
    params.inv_scale = max_abs / 127.0f;

    if (dst.dtype() == DType::U8) {
        uint8_t* dst_data = dst.data<uint8_t>();
        for (int64_t i = 0; i < n; i++) {
            float val = src_data[i] * params.scale;
            dst_data[i] = static_cast<uint8_t>(std::clamp(std::round(val), 0.0f, 255.0f));
        }
    } else if (dst.dtype() == DType::F32) {
        float* dst_data = dst.data<float>();
        for (int64_t i = 0; i < n; i++) {
            dst_data[i] = src_data[i] * params.scale;
        }
    } else {
        OIL_CHECK(false, "quantize_tensor: unsupported dst dtype");
    }

    return params;
}

std::vector<QuantizeParams> quantize_activations(const Tensor& src, Tensor& dst) {
    OIL_CHECK(src.dtype() == DType::F32, "quantize_activations: src must be F32");
    OIL_CHECK(src.rank() >= 2, "quantize_activations: src must have rank >= 2");

    int64_t batch = src.dim(0);
    int64_t per_batch = src.numel() / batch;

    std::vector<QuantizeParams> params(batch);

    for (int64_t b = 0; b < batch; b++) {
        Tensor src_slice = src.slice(0, b, b + 1);
        Tensor dst_slice = dst.slice(0, b, b + 1);
        params[b] = quantize_tensor(src_slice, dst_slice);
    }

    return params;
}

Tensor dequantize(const Tensor& src, const QuantizeParams& params) {
    int64_t n = src.numel();
    Tensor out(Shape(n), DType::F32);
    float* out_data = out.data<float>();

    if (src.dtype() == DType::U8) {
        const uint8_t* src_data = src.data<uint8_t>();
        for (int64_t i = 0; i < n; i++) {
            out_data[i] = static_cast<float>(src_data[i]) * params.inv_scale;
        }
    } else if (src.dtype() == DType::F32) {
        const float* src_data = src.data<float>();
        for (int64_t i = 0; i < n; i++) {
            out_data[i] = src_data[i] * params.inv_scale;
        }
    } else {
        OIL_CHECK(false, "dequantize: unsupported src dtype");
    }

    return out;
}

void quantize_ternary_block(const float* src, uint8_t* dst, float* scale, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = std::abs(src[i]);
        if (v > max_abs) max_abs = v;
    }

    if (max_abs == 0.0f) {
        *scale = 1.0f;
        std::memset(dst, 0, (n + 3) / 4);
        return;
    }

    float threshold = 0.577f * max_abs;
    float sum_abs = 0.0f;
    int non_zero_count = 0;
    std::vector<int> ternary(n);

    for (int i = 0; i < n; i++) {
        float v = src[i];
        if (std::abs(v) < threshold) {
            ternary[i] = 0;
        } else if (v > 0) {
            ternary[i] = 1;
            sum_abs += v;
            non_zero_count++;
        } else {
            ternary[i] = 2;
            sum_abs += -v;
            non_zero_count++;
        }
    }

    *scale = (non_zero_count > 0) ? (sum_abs / non_zero_count) : 1.0f;

    for (int i = 0; i < n; i += 4) {
        uint8_t byte = 0;
        for (int j = 0; j < 4 && (i + j) < n; j++) {
            int t = ternary[i + j];
            byte |= static_cast<uint8_t>(t) << (2 * j);
        }
        dst[i / 4] = byte;
    }
}

void quantize_binary_block(const float* src, uint8_t* dst, float* scale, int n) {
    float sum_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_abs += std::abs(src[i]);
    }
    *scale = (n > 0) ? (sum_abs / n) : 1.0f;

    for (int i = 0; i < n; i += 8) {
        uint8_t byte = 0;
        for (int j = 0; j < 8 && (i + j) < n; j++) {
            if (src[i + j] > 0) {
                byte |= static_cast<uint8_t>(1) << j;
            }
        }
        dst[i / 8] = byte;
    }
}

void dequantize_ternary_block(const uint8_t* src, float* dst, float scale, int n) {
    for (int i = 0; i < n; i += 4) {
        uint8_t byte = src[i / 4];
        for (int j = 0; j < 4 && (i + j) < n; j++) {
            int t = (byte >> (2 * j)) & 0x03;
            float val;
            if (t == 0) {
                val = 0.0f;
            } else if (t == 1) {
                val = scale;
            } else {
                val = -scale;
            }
            dst[i + j] = val;
        }
    }
}

void dequantize_binary_block(const uint8_t* src, float* dst, float scale, int n) {
    for (int i = 0; i < n; i += 8) {
        uint8_t byte = src[i / 8];
        for (int j = 0; j < 8 && (i + j) < n; j++) {
            dst[i + j] = ((byte >> j) & 1) ? scale : -scale;
        }
    }
}

} // namespace oil8
} // namespace oil
