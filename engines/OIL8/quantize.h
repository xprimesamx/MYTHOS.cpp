#pragma once
#include <cstdint>
#include <vector>
#include "oil/tensor.h"

namespace oil {
namespace oil8 {

struct QuantizeParams {
    float scale;
    float inv_scale;
};

QuantizeParams quantize_tensor(const Tensor& src, Tensor& dst);
std::vector<QuantizeParams> quantize_activations(const Tensor& src, Tensor& dst);
Tensor dequantize(const Tensor& src, const QuantizeParams& params);
void quantize_ternary_block(const float* src, uint8_t* dst, float* scale, int n);
void quantize_binary_block(const float* src, uint8_t* dst, float* scale, int n);
void dequantize_ternary_block(const uint8_t* src, float* dst, float scale, int n);
void dequantize_binary_block(const uint8_t* src, float* dst, float scale, int n);

} // namespace oil8
} // namespace oil
