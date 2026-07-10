#include "oil/kernel.h"
#include "oil/tensor.h"
#include "oil/codebook.h"
#include <cstring>
#include <cmath>
#include <cstdint>

namespace oil {
namespace kernel {

void oil8_gemm(const uint8_t* indices, const float* codebook,
               const float* activations, float* output,
               int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            const uint8_t* w_row = indices + (m * N + n) * K;
            const float* a_row = activations + n * K;
            for (int k = 0; k < K; k++) {
                uint8_t idx = w_row[k];
                float w = codebook[idx];
                sum += w * a_row[k];
            }
            output[m * N + n] = sum;
        }
    }
}

} // namespace kernel
} // namespace oil
