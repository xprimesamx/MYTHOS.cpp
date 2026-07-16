#pragma once
#include "oil/tensor.h"
#include "oil/types.h"
#include <vector>
#include <cstdint>
#include <string>

namespace oil {
namespace engines {

// FP8 E4M3: 1 sign + 4 exponent + 3 mantissa
float fp8_e4m3_dequantize(uint8_t bits);
uint8_t fp8_e4m3_quantize(float val);
Tensor fp8_e4m3_dequant_tensor(const uint8_t* data, int64_t n);

// FP8 E5M2: 1 sign + 5 exponent + 2 mantissa
float fp8_e5m2_dequantize(uint8_t bits);
uint8_t fp8_e5m2_quantize(float val);
Tensor fp8_e5m2_dequant_tensor(const uint8_t* data, int64_t n);

// NF4: Normal Float 4-bit (QLoRA)
uint8_t nf4_quantize(float val, float scale);
float nf4_dequantize(uint8_t idx, float scale);
Tensor nf4_dequant_tensor(const uint8_t* data, const float* scales,
                          int64_t n, int64_t block_size);

// AWQ: Activation-aware Weight Quantization
class AWQQuantizer {
public:
    AWQQuantizer(int64_t group_size = 128, float alpha = 0.5f);
    void compute_scales(const Tensor& weight, const Tensor& activation);
    Tensor quantize(const Tensor& weight);
    Tensor dequantize(const Tensor& q_weight);
    std::vector<float> scales_;
private:
    int64_t group_size_;
    float alpha_;
};

// GPTQ: Hessian-based quantization
class GPTQQuantizer {
public:
    GPTQQuantizer(int64_t group_size = 128, int bits = 4);
    Tensor quantize(const Tensor& weight, const Tensor& hessian);
    Tensor dequantize(const Tensor& q_weight);
private:
    int64_t group_size_;
    int bits_;
    float max_q_;
    float min_q_;
};

// I2S Engine: Int2 + Scale (BitNet compatible)
class I2SEngine {
public:
    explicit I2SEngine(int64_t block_size = 128);
    Tensor quantize(const Tensor& weight);
    Tensor dequantize(const Tensor& packed, const Tensor& scales, int64_t n);
private:
    int64_t block_size_;
};

// OIL8 Engine: 256-entry FP32 codebook
class OIL8Engine {
public:
    OIL8Engine();
    void train_codebook(const float* data, int64_t n);
    uint8_t quantize(float val) const;
    float dequantize(uint8_t idx) const;
    Tensor dequant_tensor(const uint8_t* indices, int64_t n) const;
    const std::vector<float>& codebook() const { return codebook_; }
private:
    std::vector<float> codebook_;
};

// OIL4 Engine: 16-entry FP16 codebook
class OIL4Engine {
public:
    OIL4Engine();
    void train_codebook(const float* data, int64_t n);
    uint8_t quantize(float val) const;
    float dequantize(uint8_t idx) const;
    Tensor dequant_tensor(const uint8_t* indices, int64_t n) const;
    const std::vector<float>& codebook() const { return codebook_; }
private:
    std::vector<float> codebook_;
};

// Ternary Engine: {-1, 0, +1} with per-block scale
class TernaryEngine {
public:
    explicit TernaryEngine(int64_t block_size = 128);
    Tensor quantize(const Tensor& weight);
    Tensor dequantize(const Tensor& packed, const Tensor& scales, int64_t n);
private:
    int64_t block_size_;
};

// Binary Engine: {-1, +1} with per-tensor scale
class BinaryEngine {
public:
    BinaryEngine();
    Tensor quantize(const Tensor& weight);
    Tensor dequantize(const Tensor& packed, float scale, int64_t n);
};

// Error computation
float compute_quant_error(const Tensor& original, const Tensor& dequantized);

} // namespace engines
} // namespace oil
