#include "oil/kernel.h"
#include "oil/tensor.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>

namespace oil {
namespace kernel {

// TL1 precompute: group of 2 ternary weights
// 9 possible sums {-2,-1,0,1,2} * scales
void tl1_precompute_lut(const int8_t* activations, int8_t* lut,
                        int K, float scales) {
    int8_t* lut_ptr = lut;
    for (int k = 0; k < K; k += 2) {
        int8_t a0 = activations[k];
        int8_t a1 = activations[k + 1];
        int idx = 0;
        for (int w0 = -1; w0 <= 1; w0++) {
            for (int w1 = -1; w1 <= 1; w1++) {
                int32_t sum = (int32_t)a0 * w0 + (int32_t)a1 * w1;
                lut_ptr[idx] = (int8_t)std::clamp(sum, -128, 127);
                idx++;
            }
        }
        lut_ptr += 9;
    }
}

void tl1_gemm(const Tensor& weights, const Tensor& activations,
              Tensor& output, int M, int N, int K) {
    const uint8_t* w = (const uint8_t*)weights.data();
    const float* a = (const float*)activations.data();
    
    Tensor a_int8(Shape{K}, DType::U8);
    int8_t* a_i8 = (int8_t*)a_int8.data();
    float a_scale;
    
    // Quantize activations per row
    for (int n = 0; n < N; n++) {
        float max_abs = 0;
        for (int k = 0; k < K; k++) {
            float abs_v = std::fabs(a[n * K + k]);
            if (abs_v > max_abs) max_abs = abs_v;
        }
        if (max_abs < 1e-10f) max_abs = 1e-10f;
        a_scale = 127.0f / max_abs;
        for (int k = 0; k < K; k++) {
            a_i8[n * K + k] = (int8_t)std::round(a[n * K + k] * a_scale);
        }
    }
    
    // Build LUT and compute
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            const uint8_t* w_row = w + (m * N + n) * K / 4;
            const int8_t* a_row = a_i8 + n * K;
            int32_t total = 0;
            int lut_size = K / 2;
            int8_t* lut = new int8_t[lut_size * 9];
            
            tl1_precompute_lut(a_row, lut, K, 1.0f);
            
            for (int i = 0; i < K; i += 2) {
                uint8_t packed = w_row[i / 4];
                int wi = (packed >> ((i % 4) * 2)) & 3;
                total += lut[(i / 2) * 9 + wi];
            }
            
            delete[] lut;
            ((float*)output.data())[m * N + n] = (float)total / a_scale;
        }
    }
}

// TL2 placeholder (simplified - group of 3)
void tl2_precompute_lut(const int8_t* activations, int8_t* lut,
                        int K, float scales) {
}

void tl2_gemm(const Tensor& weights, const Tensor& activations,
              Tensor& output, int M, int N, int K) {
    tl1_gemm(weights, activations, output, M, N, K);
}

} // namespace kernel
} // namespace oil
