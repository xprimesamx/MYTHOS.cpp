#include "oil/kernel.h"
#include "oil/tensor.h"
#include "oil/codebook.h"
#include <cmath>
#include <cstring>
#include <cstdint>

namespace oil {
namespace kernel {

// I2_S unpack: extract ternary values {-1,0,+1} * scale from packed 2-bit
// Storage: 4 values per byte: [v3:2][v2:2][v1:2][v0:2]
// Mapping: 00=-1, 01=0, 10=+1
static inline int decode_i2s(uint8_t packed, int shift) {
    int val = (packed >> shift) & 3;
    if (val == 0) return -1;
    if (val == 1) return 0;
    return 1;
}

void i2s_gemv(const uint8_t* packed_w, float w_scale,
              const int8_t* act, float act_scale,
              float* output, int K) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        int w = decode_i2s(packed_w[i / 4], (i % 4) * 2);
        sum += (float)w * (float)act[i];
    }
    *output = sum * w_scale * act_scale;
}

void i2s_gemm(const Tensor& weights, const Tensor& activations,
              Tensor& output, int M, int N, int K) {
    const uint8_t* w = (const uint8_t*)weights.data();
    const float* a = (const float*)activations.data();
    Tensor a_int8(Shape{K}, DType::U8);
    int8_t* a_i8 = (int8_t*)a_int8.data();
    float a_scale = 1.0f;
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k += 128) {
                int block_k = (K - k < 128) ? K - k : 128;
                const uint8_t* wb = w + (m * N + n) * K / 4 + k / 4;
                const float* ab = a + n * K + k;
                float block_sum = 0;
                for (int i = 0; i < block_k; i++) {
                    int wv = decode_i2s(wb[i / 4], (i % 4) * 2);
                    block_sum += (float)wv * ab[i];
                }
                sum += block_sum;
            }
            ((float*)output.data())[m * N + n] = sum;
        }
    }
}

void scalar_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

#if defined(OIL_AVX2)
#include <immintrin.h>
void avx2_gemm(const float* A, const float* B, float* C,
               int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n += 8) {
            __m256 cvec = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 avec = _mm256_set1_ps(A[m * K + k]);
                __m256 bvec = _mm256_loadu_ps(&B[k * N + n]);
                cvec = _mm256_fmadd_ps(avec, bvec, cvec);
            }
            _mm256_storeu_ps(&C[m * N + n], cvec);
        }
    }
}
#else
void avx2_gemm(const float* A, const float* B, float* C,
               int M, int N, int K) {
    scalar_gemm(A, B, C, M, N, K);
}
#endif

} // namespace kernel
} // namespace oil
