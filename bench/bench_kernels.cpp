#include "oil/kernel.h"
#include "oil/tensor.h"
#include "oil/types.h"

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

    return 0;
}
