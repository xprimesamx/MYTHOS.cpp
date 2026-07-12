#include "oil/kernel.h"
#include "oil/tensor.h"
#include "oil/codebook.h"
#include <cstring>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

namespace oil {
namespace kernel {

static void oil8_gemm_scalar(const uint8_t* indices, const float* codebook,
                              const float* activations, float* output,
                              int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            const uint8_t* w_row = indices + ((int64_t)m * N + n) * K;
            const float* a_row = activations + (int64_t)n * K;
            for (int k = 0; k < K; k++)
                sum += codebook[w_row[k]] * a_row[k];
            output[m * N + n] = sum;
        }
    }
}

static void oil8_gemm_avx2(const uint8_t* indices, const float* codebook,
                            const float* activations, float* output,
                            int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            __m256 sum8 = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                const uint8_t* w = indices + ((int64_t)m * N + n) * K + k;
                const float* a = activations + (int64_t)n * K + k;
                __m128i idx8 = _mm_loadl_epi64((const __m128i*)w);
                __m256i idx32 = _mm256_cvtepu8_epi32(idx8);
                __m256 wf32 = _mm256_i32gather_ps(codebook, idx32, 4);
                __m256 af = _mm256_loadu_ps(a);
                sum8 = _mm256_fmadd_ps(wf32, af, sum8);
            }
            __m128 hi = _mm256_extractf128_ps(sum8, 1);
            __m128 lo = _mm256_castps256_ps128(sum8);
            __m128 sum4 = _mm_add_ps(lo, hi);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            float result = _mm_cvtss_f32(sum4);
            for (; k < K; k++) {
                const uint8_t* w = indices + ((int64_t)m * N + n) * K + k;
                const float* a = activations + (int64_t)n * K + k;
                result += codebook[*w] * *a;
            }
            output[m * N + n] = result;
        }
    }
}

void oil8_gemm(const uint8_t* indices, const float* codebook,
               const float* activations, float* output,
               int M, int N, int K) {
#if defined(__AVX2__)
    oil8_gemm_avx2(indices, codebook, activations, output, M, N, K);
#else
    oil8_gemm_scalar(indices, codebook, activations, output, M, N, K);
#endif
}

} // namespace kernel
} // namespace oil
