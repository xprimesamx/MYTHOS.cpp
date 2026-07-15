#include "oil/kernel.h"
#include "oil/tensor.h"
#include "oil/types.h"
#include "oil/math.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

struct BenchResult {
    std::string name;
    int M, N, K;
    double time_us;
    double gflops;
};

static BenchResult bench_gemm(const std::string& name,
                               void (*gemm_fn)(const float*, const float*, float*, int, int, int),
                               int M, int N, int K) {
    oil::Tensor A(oil::Shape{M, K}, oil::DType::F32);
    oil::Tensor B(oil::Shape{K, N}, oil::DType::F32);
    oil::Tensor C(oil::Shape{M, N}, oil::DType::F32);

    A.fill(1.0f);
    B.fill(2.0f);

    int warmup = 5;
    for (int i = 0; i < warmup; i++)
        gemm_fn(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);

    int iters = 50;
    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        gemm_fn(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);
    double dt = (now_sec() - t0) / iters;

    double flops = 2.0 * (double)M * (double)N * (double)K;
    return {name, M, N, K, dt * 1e6, flops / dt * 1e-9};
}

// ---------------------------------------------------------------------------
// bench_tiled_gemm: compare all available GEMM variants on a fixed size
// ---------------------------------------------------------------------------
static void bench_tiled_gemm() {
    std::cout << "\n=== Tiled GEMM Comparison (256x256x256) ===" << std::endl;

    const int M = 256, N = 256, K = 256;
    struct { const char* name; void (*fn)(const float*, const float*, float*, int, int, int); } variants[] = {
        {"scalar_gemm",  oil::kernel::scalar_gemm},
        {"tiled_gemm",   oil::kernel::tiled_gemm},
#if defined(OIL_AVX2)
        {"avx2_gemm",    oil::kernel::avx2_gemm},
        {"avx2_tiled",   oil::kernel::avx2_tiled_gemm},
#endif
    };
    int nv = sizeof(variants) / sizeof(variants[0]);
    std::vector<BenchResult> results;

    for (int i = 0; i < nv; i++) {
        auto r = bench_gemm(variants[i].name, variants[i].fn, M, N, K);
        results.push_back(r);
        std::cout << std::left << std::setw(16) << r.name
                  << " time: " << std::fixed << std::setprecision(2) << r.time_us << " us"
                  << "  GFLOPS: " << std::fixed << std::setprecision(2) << r.gflops << std::endl;
    }

    // Speedup vs scalar
    if (results.size() >= 2) {
        double base = results[0].time_us;
        for (size_t i = 1; i < results.size(); i++) {
            std::cout << "  Speedup " << results[i].name << ": "
                      << std::fixed << std::setprecision(2) << (base / results[i].time_us) << "x" << std::endl;
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar reference softmax (used when OIL_AVX2 is not defined, for comparison)
// ---------------------------------------------------------------------------
static void scalar_softmax_ref(const float* src, float* dst, int64_t dim) {
    float mx = -INFINITY;
    for (int64_t i = 0; i < dim; i++) mx = std::max(mx, src[i]);
    double sum = 0.0;
    for (int64_t i = 0; i < dim; i++) { float e = std::exp(src[i] - mx); dst[i] = e; sum += e; }
    double inv = 1.0 / sum;
    for (int64_t i = 0; i < dim; i++) dst[i] = (float)((double)dst[i] * inv);
}

// ---------------------------------------------------------------------------
// bench_softmax: compare scalar reference vs oil::math::softmax
// ---------------------------------------------------------------------------
static void bench_softmax() {
    std::cout << "\n=== Softmax Benchmarks ===" << std::endl;

    std::vector<int64_t> dims = {64, 128, 256, 512, 1024, 4096};
    const int rows = 100;

    for (int64_t dim : dims) {
        oil::Tensor x(oil::Shape{rows, dim}, oil::DType::F32);
        oil::Tensor y(oil::Shape{rows, dim}, oil::DType::F32);
        x.fill(1.0f);

        // Warmup
        for (int i = 0; i < 5; i++)
            scalar_softmax_ref(x.data<float>(), y.data<float>(), dim);

        int iters = 50;
        double t0 = now_sec();
        for (int i = 0; i < iters; i++) {
            for (int r = 0; r < rows; r++)
                scalar_softmax_ref(x.data<float>() + r * dim, y.data<float>() + r * dim, dim);
        }
        double scalar_us = (now_sec() - t0) / iters * 1e6;

        // Warmup
        for (int i = 0; i < 5; i++)
            oil::math::softmax(x, y, 1);

        t0 = now_sec();
        for (int i = 0; i < iters; i++)
            oil::math::softmax(x, y, 1);
        double vec_us = (now_sec() - t0) / iters * 1e6;

        std::cout << "  dim=" << std::setw(5) << dim
                  << "  scalar: " << std::fixed << std::setprecision(2) << scalar_us << " us"
                  << "  math::softmax: " << std::fixed << std::setprecision(2) << vec_us << " us"
                  << "  (" << std::fixed << std::setprecision(2) << (scalar_us / vec_us) << "x)" << std::endl;
    }
}

// ---------------------------------------------------------------------------
// bench_attention: attention score Q @ K^T / sqrt(D) + softmax
// K is stored transposed as [head_dim, S] so GEMM interprets it as K^T naturally
// ---------------------------------------------------------------------------
static void bench_attention() {
    std::cout << "\n=== Attention Score Benchmarks ===" << std::endl;

    std::vector<int> seq_lens = {64, 128, 256, 512};
    const int head_dim = 64;
    const int num_heads = 8;
    const int iters = 20;

    for (int S : seq_lens) {
        // Q: [S, head_dim], K^T: [head_dim, S]
        oil::Tensor Q(oil::Shape{num_heads, S, head_dim}, oil::DType::F32);
        oil::Tensor KT(oil::Shape{num_heads, head_dim, S}, oil::DType::F32);
        oil::Tensor score(oil::Shape{num_heads, S, S}, oil::DType::F32);
        Q.fill(1.0f); KT.fill(1.0f);

        double scale = 1.0 / std::sqrt((double)head_dim);

        // Scalar attention: naive Q @ K^T using K stored transposed
        auto t0 = now_sec();
        for (int iter = 0; iter < iters; iter++) {
            for (int h = 0; h < num_heads; h++) {
                const float* q = Q.data<float>() + h * S * head_dim;
                const float* kt = KT.data<float>() + h * head_dim * S;
                float* s = score.data<float>() + h * S * S;
                for (int i = 0; i < S; i++) {
                    for (int j = 0; j < S; j++) {
                        float dot = 0;
                        for (int d = 0; d < head_dim; d++)
                            dot += q[i * head_dim + d] * kt[d * S + j];
                        s[i * S + j] = (float)(dot * scale);
                    }
                }
                for (int i = 0; i < S; i++)
                    scalar_softmax_ref(s + i * S, s + i * S, S);
            }
        }
        double scalar_us = (now_sec() - t0) / iters * 1e6;

        // GEMM-based attention: scalar_gemm(Q, K^T, score, S, S, head_dim) + softmax
        t0 = now_sec();
        for (int iter = 0; iter < iters; iter++) {
            for (int h = 0; h < num_heads; h++) {
                const float* q = Q.data<float>() + h * S * head_dim;
                const float* kt = KT.data<float>() + h * head_dim * S;
                float* s = score.data<float>() + h * S * S;
                oil::kernel::scalar_gemm(q, kt, s, S, S, head_dim);
                for (int64_t i = 0; i < (int64_t)S * S; i++) s[i] *= (float)scale;
                oil::Tensor head_score(oil::Shape{S, S}, oil::DType::F32);
                std::memcpy(head_score.data<float>(), s, S * S * sizeof(float));
                oil::math::softmax(head_score, head_score, 1);
                std::memcpy(s, head_score.data<float>(), S * S * sizeof(float));
            }
        }
        double gemm_us = (now_sec() - t0) / iters * 1e6;

        std::cout << "  S=" << std::setw(4) << S
                  << "  scalar: " << std::fixed << std::setprecision(2) << scalar_us << " us"
                  << "  gemm:   " << std::fixed << std::setprecision(2) << gemm_us << " us"
                  << "  (" << std::fixed << std::setprecision(2) << (scalar_us / gemm_us) << "x)" << std::endl;
    }
}

int main() {
    std::cout << "=== OIL Kernel Benchmarks ===" << std::endl;
    std::cout << std::left << std::setw(20) << "Kernel"
              << std::setw(12) << "Size"
              << std::setw(14) << "Time (us)"
              << std::setw(14) << "GFLOPS" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    std::vector<int> sizes = {128, 256, 512, 1024};

    for (int sz : sizes) {
        auto r = bench_gemm("scalar_gemm", oil::kernel::scalar_gemm, sz, sz, sz);
        std::cout << std::left << std::setw(20) << r.name
                  << std::setw(12) << (std::to_string(r.M) + "x" + std::to_string(r.N) + "x" + std::to_string(r.K))
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.time_us
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.gflops << std::endl;
    }

#if defined(OIL_AVX2)
    std::cout << "\n--- AVX2 ---" << std::endl;
    for (int sz : sizes) {
        auto r = bench_gemm("avx2_gemm", oil::kernel::avx2_gemm, sz, sz, sz);
        std::cout << std::left << std::setw(20) << r.name
                  << std::setw(12) << (std::to_string(r.M) + "x" + std::to_string(r.N) + "x" + std::to_string(r.K))
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.time_us
                  << std::setw(14) << std::fixed << std::setprecision(2) << r.gflops << std::endl;
    }

    // Speedup comparison
    std::cout << "\n=== Speedup (AVX2 vs Scalar) ===" << std::endl;
    for (int sz : sizes) {
        auto r_scalar = bench_gemm("scalar_gemm", oil::kernel::scalar_gemm, sz, sz, sz);
        auto r_avx2 = bench_gemm("avx2_gemm", oil::kernel::avx2_gemm, sz, sz, sz);
        double speedup = r_scalar.time_us / r_avx2.time_us;
        std::cout << "Size " << sz << "x" << sz << "x" << sz
                  << ": " << std::fixed << std::setprecision(2) << speedup << "x speedup" << std::endl;
    }
#else
    std::cout << "\nAVX2 not available. Build with -mavx2 -mfma -DOIL_AVX2 for AVX2 benchmarks." << std::endl;
#endif

    // MFLOPS summary for largest size
    std::cout << "\n=== MFLOPS Summary (size=1024) ===" << std::endl;
    auto r1024 = bench_gemm("scalar_gemm", oil::kernel::scalar_gemm, 1024, 1024, 1024);
    std::cout << "scalar_gemm: " << std::fixed << std::setprecision(0) << (r1024.gflops * 1000) << " MFLOPS" << std::endl;

    // Additional benchmark suites
    bench_tiled_gemm();
    bench_softmax();
    bench_attention();

    std::cout << "\nDone." << std::endl;
    return 0;
}
