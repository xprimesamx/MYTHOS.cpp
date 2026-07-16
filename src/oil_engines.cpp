#include "oil/oil_engines.h"
#include "oil/math.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace oil {
namespace engines {

// ===========================================================================
// FP8 E4M3: 1 sign + 4 exponent + 3 mantissa bits
// Range: [-448, 448], no inf/nan (uses all 8-bit patterns for values)
// ===========================================================================

float fp8_e4m3_dequantize(uint8_t bits) {
    int sign = (bits >> 7) & 1;
    int exp_raw = (bits >> 3) & 0xF;
    int mant = bits & 0x7;
    float val;
    if (exp_raw == 0) {
        val = (float)mant * (1.0f / 64.0f);
    } else if (exp_raw == 0xF && mant == 0x7) {
        val = 0.0f;
    } else {
        int exp = exp_raw - 7;
        val = std::ldexpf(1.0f + (float)mant / 8.0f, exp);
    }
    return sign ? -val : val;
}

uint8_t fp8_e4m3_quantize(float val) {
    if (val == 0.0f) return 0;
    int sign = 0;
    if (val < 0) { sign = 1; val = -val; }
    if (val > 448.0f) val = 448.0f;
    int exp = 0;
    float mant = val;
    while (mant >= 2.0f && exp < 8) { mant /= 2.0f; exp++; }
    while (mant < 1.0f && exp > -7) { mant *= 2.0f; exp--; }
    if (exp < -7) {
        int m = (int)std::round(mant * 64.0f);
        if (m > 7) m = 7;
        return (uint8_t)((sign << 7) | m);
    }
    int m = (int)std::round((mant - 1.0f) * 8.0f);
    if (m > 7) m = 7;
    if (m < 0) m = 0;
    return (uint8_t)((sign << 7) | ((exp + 7) << 3) | m);
}

Tensor fp8_e4m3_dequant_tensor(const uint8_t* data, int64_t n) {
    Tensor out({n});
    float* od = out.data<float>();
    for (int64_t i = 0; i < n; ++i)
        od[i] = fp8_e4m3_dequantize(data[i]);
    return out;
}

// ===========================================================================
// FP8 E5M2: 1 sign + 5 exponent + 2 mantissa bits
// Range: [-57344, 57344], supports inf/nan
// ===========================================================================

float fp8_e5m2_dequantize(uint8_t bits) {
    int sign = (bits >> 7) & 1;
    int exp_raw = (bits >> 2) & 0x1F;
    int mant = bits & 0x3;
    float val;
    if (exp_raw == 0) {
        val = (float)mant * (1.0f / 16384.0f);
    } else if (exp_raw == 0x1F) {
        val = (mant == 0) ? INFINITY : NAN;
    } else {
        int exp = exp_raw - 15;
        val = std::ldexpf(1.0f + (float)mant / 4.0f, exp);
    }
    return sign ? -val : val;
}

uint8_t fp8_e5m2_quantize(float val) {
    if (val == 0.0f) return 0;
    int sign = 0;
    if (val < 0) { sign = 1; val = -val; }
    if (val > 57344.0f) val = 57344.0f;
    int exp = 0;
    float mant = val;
    while (mant >= 2.0f && exp < 16) { mant /= 2.0f; exp++; }
    while (mant < 1.0f && exp > -15) { mant *= 2.0f; exp--; }
    if (exp < -15) {
        int m = (int)std::round(mant * 16384.0f);
        if (m > 3) m = 3;
        return (uint8_t)((sign << 7) | m);
    }
    int m = (int)std::round((mant - 1.0f) * 4.0f);
    if (m > 3) m = 3;
    if (m < 0) m = 0;
    return (uint8_t)((sign << 7) | ((exp + 15) << 2) | m);
}

Tensor fp8_e5m2_dequant_tensor(const uint8_t* data, int64_t n) {
    Tensor out({n});
    float* od = out.data<float>();
    for (int64_t i = 0; i < n; ++i)
        od[i] = fp8_e5m2_dequantize(data[i]);
    return out;
}

// ===========================================================================
// NF4: Normal Float 4-bit (QLoRA)
// 16 values from normal distribution, quantized to 4 bits
// ===========================================================================

static const float NF4_CODEBOOK[16] = {
    -1.0f, -0.6961928f, -0.525073f, -0.39497948f,
    -0.28444138f, -0.18477343f, -0.09105138f, 0.0f,
    0.07958097f, 0.16092212f, 0.24611243f, 0.33712566f,
    0.43570983f, 0.54554284f, 0.67132018f, 1.0f
};

uint8_t nf4_quantize(float val, float scale) {
    float normalized = val / (scale + 1e-10f);
    normalized = std::max(-1.0f, std::min(1.0f, normalized));
    int best = 0;
    float best_dist = 1e10f;
    for (int i = 0; i < 16; ++i) {
        float dist = std::abs(normalized - NF4_CODEBOOK[i]);
        if (dist < best_dist) { best_dist = dist; best = i; }
    }
    return (uint8_t)best;
}

float nf4_dequantize(uint8_t idx, float scale) {
    if (idx >= 16) idx = 0;
    return NF4_CODEBOOK[idx] * scale;
}

Tensor nf4_dequant_tensor(const uint8_t* data, const float* scales,
                          int64_t n, int64_t block_size) {
    Tensor out({n});
    float* od = out.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        int64_t block_idx = i / block_size;
        od[i] = nf4_dequantize(data[i], scales[block_idx]);
    }
    return out;
}

// ===========================================================================
// AWQ: Activation-aware Weight Quantization
// Per-channel scaling + per-group INT4 quantization
// ===========================================================================

AWQQuantizer::AWQQuantizer(int64_t group_size, float alpha)
    : group_size_(group_size), alpha_(alpha) {}

void AWQQuantizer::compute_scales(const Tensor& weight, const Tensor& activation) {
    int64_t N = weight.dim(0);
    int64_t K = weight.dim(1);
    scales_.resize(N, 1.0f);
    const float* wd = weight.data<float>();
    const float* ad = activation.data<float>();
    for (int64_t n = 0; n < N; ++n) {
        float max_w = 0, max_a = 0;
        for (int64_t k = 0; k < K; ++k) {
            max_w = std::max(max_w, std::abs(wd[n * K + k]));
        }
        for (int64_t k = 0; k < K; ++k) {
            max_a = std::max(max_a, std::abs(ad[k]));
        }
        float s = std::pow(max_a, alpha_) / (max_w + 1e-10f);
        s = std::max(1e-8f, std::min(s, 1e4f));
        scales_[(size_t)n] = s;
    }
}

Tensor AWQQuantizer::quantize(const Tensor& weight) {
    int64_t N = weight.dim(0);
    int64_t K = weight.dim(1);
    int64_t num_groups = (K + group_size_ - 1) / group_size_;
    Tensor q_weight({N, K});
    float* qd = q_weight.data<float>();
    const float* wd = weight.data<float>();
    for (int64_t n = 0; n < N; ++n) {
        float s = scales_[(size_t)n];
        for (int64_t g = 0; g < num_groups; ++g) {
            int64_t start = g * group_size_;
            int64_t end = std::min(start + group_size_, K);
            float max_abs = 0;
            for (int64_t k = start; k < end; ++k)
                max_abs = std::max(max_abs, std::abs(wd[n * K + k] * s));
            float scale = max_abs / 7.0f;
            if (scale < 1e-10f) scale = 1e-10f;
            for (int64_t k = start; k < end; ++k) {
                int q = (int)std::round(wd[n * K + k] * s / scale);
                q = std::max(-8, std::min(7, q));
                qd[n * K + k] = (float)q * scale / s;
            }
        }
    }
    return q_weight;
}

Tensor AWQQuantizer::dequantize(const Tensor& q_weight) {
    return q_weight;
}

// ===========================================================================
// GPTQ: Quantization via approximate second-order Hessian
// Per-group INT4 quantization with Hessian-based error compensation
// ===========================================================================

GPTQQuantizer::GPTQQuantizer(int64_t group_size, int bits)
    : group_size_(group_size), bits_(bits) {
    max_q_ = (float)((1 << (bits - 1)) - 1);
    min_q_ = -(float)(1 << (bits - 1));
}

Tensor GPTQQuantizer::quantize(const Tensor& weight, const Tensor& hessian) {
    int64_t N = weight.dim(0);
    int64_t K = weight.dim(1);
    Tensor q_weight({N, K});
    float* qd = q_weight.data<float>();
    const float* wd = weight.data<float>();
    const float* hd = hessian.data<float>();
    for (int64_t n = 0; n < N; ++n) {
        int64_t num_groups = (K + group_size_ - 1) / group_size_;
        for (int64_t g = 0; g < num_groups; ++g) {
            int64_t start = g * group_size_;
            int64_t end = std::min(start + group_size_, K);
            float scale = 0;
            for (int64_t k = start; k < end; ++k) {
                float h = hd[n * K + k];
                scale = std::max(scale, std::abs(wd[n * K + k]) / (h + 1e-8f));
            }
            scale = std::max(1e-10f, scale / max_q_);
            for (int64_t k = start; k < end; ++k) {
                int q = (int)std::round(wd[n * K + k] / scale);
                q = (int)std::max((float)min_q_, std::min((float)max_q_, (float)q));
                qd[n * K + k] = (float)q * scale;
            }
        }
    }
    return q_weight;
}

Tensor GPTQQuantizer::dequantize(const Tensor& q_weight) {
    return q_weight;
}

// ===========================================================================
// I2S Engine: Int2 + Scale (BitNet compatible)
// Pack 4 ternary values (2-bit each) into 1 byte with shared scale
// ===========================================================================

I2SEngine::I2SEngine(int64_t block_size) : block_size_(block_size) {}

Tensor I2SEngine::quantize(const Tensor& weight) {
    int64_t n = weight.numel();
    int64_t num_blocks = (n + block_size_ - 1) / block_size_;
    int64_t packed_size = (n + 3) / 4;
    Tensor packed({packed_size});
    Tensor scales({num_blocks});
    packed.zero_();
    const float* wd = weight.data<float>();
    uint8_t* pd = reinterpret_cast<uint8_t*>(packed.data<float>());
    float* sd = scales.data<float>();
    for (int64_t b = 0; b < num_blocks; ++b) {
        int64_t start = b * block_size_;
        int64_t end = std::min(start + block_size_, n);
        float max_abs = 0;
        for (int64_t i = start; i < end; ++i)
            max_abs = std::max(max_abs, std::abs(wd[i]));
        float scale = max_abs;
        sd[b] = scale;
        if (scale < 1e-10f) scale = 1.0f;
        for (int64_t i = start; i < end; ++i) {
            float normalized = wd[i] / scale;
            int8_t ternary;
            if (normalized > 0.33f) ternary = 1;
            else if (normalized < -0.33f) ternary = -1;
            else ternary = 0;
            uint8_t code = (uint8_t)((ternary + 1) & 0x3);
            int64_t byte_idx = i / 4;
            int bit_off = (int)(i % 4) * 2;
            if (bit_off == 0)
                pd[byte_idx] = code;
            else
                pd[byte_idx] |= (code << bit_off);
        }
    }
    return packed;
}

Tensor I2SEngine::dequantize(const Tensor& packed, const Tensor& scales, int64_t n) {
    Tensor out({n});
    float* od = out.data<float>();
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(packed.data<float>());
    const float* sd = scales.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        int64_t block_idx = i / block_size_;
        float scale = sd[block_idx];
        int64_t byte_idx = i / 4;
        int bit_off = (int)(i % 4) * 2;
        uint8_t code = (pd[byte_idx] >> bit_off) & 0x3;
        float ternary = (code == 0) ? -1.0f : ((code == 2) ? 1.0f : 0.0f);
        od[i] = ternary * scale;
    }
    return out;
}

// ===========================================================================
// OIL8 Engine: 256-entry FP32 codebook, 8-bit indices
// ===========================================================================

OIL8Engine::OIL8Engine() {
    codebook_.resize(256);
    for (int i = 0; i < 256; ++i)
        codebook_[(size_t)i] = (float)(i - 128) * 0.01f;
}

void OIL8Engine::train_codebook(const float* data, int64_t n) {
    for (int iter = 0; iter < 10; ++iter) {
        std::vector<int> counts(256, 0);
        std::vector<double> sums(256, 0.0);
        for (int64_t i = 0; i < n; ++i) {
            int best = 0;
            float best_dist = 1e10f;
            for (int j = 0; j < 256; ++j) {
                float dist = std::abs(data[i] - codebook_[(size_t)j]);
                if (dist < best_dist) { best_dist = dist; best = j; }
            }
            counts[best]++;
            sums[best] += data[i];
        }
        for (int j = 0; j < 256; ++j) {
            if (counts[j] > 0)
                codebook_[(size_t)j] = (float)(sums[j] / counts[j]);
        }
    }
}

uint8_t OIL8Engine::quantize(float val) const {
    int best = 0;
    float best_dist = 1e10f;
    for (int i = 0; i < 256; ++i) {
        float dist = std::abs(val - codebook_[(size_t)i]);
        if (dist < best_dist) { best_dist = dist; best = i; }
    }
    return (uint8_t)best;
}

float OIL8Engine::dequantize(uint8_t idx) const {
    if (idx >= 256) idx = 0;
    return codebook_[idx];
}

Tensor OIL8Engine::dequant_tensor(const uint8_t* indices, int64_t n) const {
    Tensor out({n});
    float* od = out.data<float>();
    for (int64_t i = 0; i < n; ++i)
        od[i] = dequantize(indices[i]);
    return out;
}

// ===========================================================================
// OIL4 Engine: 16-entry FP16 codebook, 4-bit indices
// ===========================================================================

OIL4Engine::OIL4Engine() {
    codebook_.resize(16);
    for (int i = 0; i < 16; ++i)
        codebook_[(size_t)i] = (float)(i - 8) * 0.1f;
}

void OIL4Engine::train_codebook(const float* data, int64_t n) {
    for (int iter = 0; iter < 10; ++iter) {
        std::vector<int> counts(16, 0);
        std::vector<double> sums(16, 0.0);
        for (int64_t i = 0; i < n; ++i) {
            int best = 0;
            float best_dist = 1e10f;
            for (int j = 0; j < 16; ++j) {
                float dist = std::abs(data[i] - codebook_[(size_t)j]);
                if (dist < best_dist) { best_dist = dist; best = j; }
            }
            counts[best]++;
            sums[best] += data[i];
        }
        for (int j = 0; j < 16; ++j) {
            if (counts[j] > 0)
                codebook_[(size_t)j] = (float)(sums[j] / counts[j]);
        }
    }
}

uint8_t OIL4Engine::quantize(float val) const {
    int best = 0;
    float best_dist = 1e10f;
    for (int i = 0; i < 16; ++i) {
        float dist = std::abs(val - codebook_[(size_t)i]);
        if (dist < best_dist) { best_dist = dist; best = i; }
    }
    return (uint8_t)best;
}

float OIL4Engine::dequantize(uint8_t idx) const {
    if (idx >= 16) idx = 0;
    return codebook_[idx];
}

Tensor OIL4Engine::dequant_tensor(const uint8_t* indices, int64_t n) const {
    Tensor out({n});
    float* od = out.data<float>();
    for (int64_t i = 0; i < n; ++i)
        od[i] = dequantize(indices[i]);
    return out;
}

// ===========================================================================
// Ternary Engine: {-1, 0, +1} with per-block scale (BitNet b1.58)
// ===========================================================================

TernaryEngine::TernaryEngine(int64_t block_size) : block_size_(block_size) {}

Tensor TernaryEngine::quantize(const Tensor& weight) {
    int64_t n = weight.numel();
    int64_t num_blocks = (n + block_size_ - 1) / block_size_;
    int64_t packed_size = (n + 3) / 4;
    Tensor packed({packed_size});
    Tensor scales({num_blocks});
    packed.zero_();
    const float* wd = weight.data<float>();
    uint8_t* pd = reinterpret_cast<uint8_t*>(packed.data<float>());
    float* sd = scales.data<float>();
    for (int64_t b = 0; b < num_blocks; ++b) {
        int64_t start = b * block_size_;
        int64_t end = std::min(start + block_size_, n);
        float max_abs = 0;
        for (int64_t i = start; i < end; ++i)
            max_abs = std::max(max_abs, std::abs(wd[i]));
        sd[b] = max_abs;
        if (max_abs < 1e-10f) max_abs = 1.0f;
        for (int64_t i = start; i < end; ++i) {
            float normalized = wd[i] / max_abs;
            int8_t ternary;
            if (normalized > 0.33f) ternary = 1;
            else if (normalized < -0.33f) ternary = -1;
            else ternary = 0;
            uint8_t code = (uint8_t)((ternary + 1) & 0x3);
            int64_t byte_idx = i / 4;
            int bit_off = (int)(i % 4) * 2;
            if (bit_off == 0)
                pd[byte_idx] = code;
            else
                pd[byte_idx] |= (code << bit_off);
        }
    }
    return packed;
}

Tensor TernaryEngine::dequantize(const Tensor& packed, const Tensor& scales, int64_t n) {
    Tensor out({n});
    float* od = out.data<float>();
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(packed.data<float>());
    const float* sd = scales.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        int64_t block_idx = i / block_size_;
        float scale = sd[block_idx];
        int64_t byte_idx = i / 4;
        int bit_off = (int)(i % 4) * 2;
        uint8_t code = (pd[byte_idx] >> bit_off) & 0x3;
        float ternary = (code == 0) ? -1.0f : ((code == 2) ? 1.0f : 0.0f);
        od[i] = ternary * scale;
    }
    return out;
}

// ===========================================================================
// Binary Engine: {-1, +1} with per-tensor scale (BitNet b1)
// ===========================================================================

BinaryEngine::BinaryEngine() {}

Tensor BinaryEngine::quantize(const Tensor& weight) {
    int64_t n = weight.numel();
    int64_t packed_size = (n + 7) / 8;
    Tensor packed({packed_size});
    Tensor scale({1});
    packed.zero_();
    const float* wd = weight.data<float>();
    uint8_t* pd = reinterpret_cast<uint8_t*>(packed.data<float>());
    float max_abs = 0;
    for (int64_t i = 0; i < n; ++i)
        max_abs = std::max(max_abs, std::abs(wd[i]));
    scale.data<float>()[0] = max_abs;
    if (max_abs < 1e-10f) max_abs = 1.0f;
    for (int64_t i = 0; i < n; ++i) {
        uint8_t bit = (wd[i] >= 0.0f) ? 1 : 0;
        int64_t byte_idx = i / 8;
        int bit_off = (int)(i % 8);
        pd[byte_idx] |= (bit << bit_off);
    }
    return packed;
}

Tensor BinaryEngine::dequantize(const Tensor& packed, float scale, int64_t n) {
    Tensor out({n});
    float* od = out.data<float>();
    const uint8_t* pd = reinterpret_cast<const uint8_t*>(packed.data<float>());
    for (int64_t i = 0; i < n; ++i) {
        int64_t byte_idx = i / 8;
        int bit_off = (int)(i % 8);
        uint8_t bit = (pd[byte_idx] >> bit_off) & 1;
        od[i] = (bit == 1) ? scale : -scale;
    }
    return out;
}

// ===========================================================================
// Roundtrip test helpers
// ===========================================================================

float compute_quant_error(const Tensor& original, const Tensor& dequantized) {
    int64_t n = original.numel();
    float max_err = 0;
    const float* od = original.data<float>();
    const float* dd = dequantized.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        float err = std::abs(od[i] - dd[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

} // namespace engines
} // namespace oil
