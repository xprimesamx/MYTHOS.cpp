#include "oil/kernel.h"
#include "oil/tensor.h"
#include "oil/codebook.h"
#include <cstring>
#include <cmath>
#include <cstdint>

namespace oil {
namespace kernel {

// FP16 bits -> float
static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        // Subnormal
        float v = (float)mant * 0.000000059604644775390625f;
        return sign ? -v : v;
    } else if (exp == 31) {
        return (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
    }
    uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

void oil4_gemm(const uint8_t* packed_indices, const uint16_t* codebook,
               const float* activations, float* output,
               int M, int N, int K) {
    float f16_centroids[16];
    for (int i = 0; i < 16; i++) {
        f16_centroids[i] = fp16_to_float(codebook[i]);
    }
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            const uint8_t* w_row = packed_indices + (m * N + n) * K / 2;
            const float* a_row = activations + n * K;
            for (int k = 0; k < K; k++) {
                uint8_t idx;
                if (k % 2 == 0) {
                    idx = (w_row[k / 2] >> 4) & 0xF;
                } else {
                    idx = w_row[k / 2] & 0xF;
                }
                float w = f16_centroids[idx];
                sum += w * a_row[k];
            }
            output[m * N + n] = sum;
        }
    }
}

} // namespace kernel
} // namespace oil
