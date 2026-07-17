#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include "oil/backend.h"
#include "oil/tensor.h"
#include "oil/math.h"
#include "oil/kernel.h"
#include "oil/gpu_compute.h"

using namespace oil;
using namespace oil::backend;

int tests_passed = 0;
int tests_total = 0;

#define CHECK(cond, msg) do { \
    tests_total++; \
    if (!(cond)) { printf("  FAIL: %s\n", msg); } \
    else { tests_passed++; } \
} while(0)

static void test_hardware_probe() {
    printf("\n=== HARDWARE PROBE ===\n");
    HardwareProfile hw = probe_hardware();

    printf("  CPU: %d cores, %d threads, AVX2=%d AVX512=%d NEON=%d\n",
           (int)hw.cpu_cores, (int)hw.cpu_threads,
           hw.has_avx2, hw.has_avx512, hw.has_neon);
    printf("  GPU: DirectX=%d CUDA=%d Vulkan=%d\n",
           hw.has_directx, hw.has_cuda, hw.has_vulkan);
    printf("  RAM: total=%.1f GB free=%.1f GB\n",
           hw.ram_total / 1e9, hw.ram_free / 1e9);
    printf("  VRAM: total=%.1f GB free=%.1f GB\n",
           hw.vram_total / 1e9, hw.vram_free / 1e9);
    printf("  OS: Windows=%d Linux=%d macOS=%d ARM=%d\n",
           hw.is_windows, hw.is_linux, hw.is_macos, hw.is_arm);

    CHECK(hw.cpu_threads > 0, "cpu_threads > 0");
    CHECK(hw.ram_total > 0, "ram_total > 0");

    // Auto-select backend
    BackendConfig cfg = select_optimal_backend(hw, 0);
    printf("  Selected backend: %s\n", backend_name(cfg.type));
    CHECK(cfg.type != BackendType::CPU_SCALAR || !hw.has_avx2,
          "preferred backend when AVX2 available");
}

static void test_backend_creation() {
    printf("\n=== BACKEND CREATION ===\n");
    HardwareProfile hw = probe_hardware(); (void)hw;

    // Try creating each available backend
    BackendType types[] = {
        BackendType::CPU_SCALAR,
        BackendType::CPU_AVX2,
        BackendType::GPU_DIRECTX,
        BackendType::IGPU_SHARED,
        BackendType::RAM_SWAP
    };

    for (auto type : types) {
        BackendConfig cfg;
        cfg.type = type;
        ComputeBackend* bk = ComputeBackend::create(cfg);
        if (bk) {
            printf("  %s: available=%d\n", backend_name(type), bk->is_available());
            CHECK(bk->is_available() || type == BackendType::CPU_SCALAR,
                  "backend created");
            delete bk;
        }
    }
}

static void test_bench_gemm() {
    printf("\n=== BENCH GEMM ===\n");
    HardwareProfile hw = probe_hardware();

    struct BenchRun {
        BackendType type;
        const char* name;
        int64_t M, N, K;
    };

    // Scalar: smaller sizes (naive O(N³) is slow for large)
    BenchRun runs[] = {
        {BackendType::CPU_SCALAR, "CPU_SCALAR", 128, 128, 128},
        {BackendType::CPU_SCALAR, "CPU_SCALAR", 256, 256, 256},
    };

    for (auto& r : runs) {
        BackendConfig cfg;
        cfg.type = r.type;
        ComputeBackend* bk = ComputeBackend::create(cfg);
        if (bk && bk->is_available()) {
            double gflops = benchmark_operation(bk, "gemm", r.M, r.N, r.K, 3, 10);
            printf("  %s gemm %lldx%lldx%lld: %.2f GFLOPS\n",
                   r.name, (long long)r.M, (long long)r.N, (long long)r.K, gflops);
            CHECK(gflops > 0, "scalar gemm benchmark > 0 GFLOPS");
        }
        delete bk;
    }

    // AVX2: can handle larger, but still not tiled
    if (hw.has_avx2) {
        BenchRun avx2_runs[] = {
            {BackendType::CPU_AVX2, "CPU_AVX2", 256, 256, 256},
            {BackendType::CPU_AVX2, "CPU_AVX2", 512, 512, 512},
        };
        for (auto& r : avx2_runs) {
            BackendConfig cfg;
            cfg.type = r.type;
            ComputeBackend* bk = ComputeBackend::create(cfg);
            if (bk && bk->is_available()) {
                double gflops = benchmark_operation(bk, "gemm", r.M, r.N, r.K, 3, 10);
                printf("  %s gemm %lldx%lldx%lld: %.2f GFLOPS\n",
                       r.name, (long long)r.M, (long long)r.N, (long long)r.K, gflops);
                CHECK(gflops > 0, "avx2 gemm benchmark > 0 GFLOPS");
            }
            delete bk;
        }
    }
}

static void test_bench_operations() {
    printf("\n=== BENCH OPERATIONS ===\n");
    const int64_t N = 4096;

    BackendConfig cfg;
    cfg.type = BackendType::CPU_SCALAR;
    ComputeBackend* bk = ComputeBackend::create(cfg);
    if (bk && bk->is_available()) {
        struct Op { const char* name; const char* op; int64_t M, N2, K; };
        Op ops[] = {
            {"gemm", "gemm", 256, 256, 256},
            {"relu", "relu", N, N, 0},
            {"add",  "add",  N, N, 0},
        };
        for (auto& o : ops) {
            double gflops = benchmark_operation(bk, o.op, o.M, o.N2, o.K, 5, 30);
            printf("  SCALAR %s: %.2f GFLOPS\n", o.name, gflops);
            CHECK(gflops > 0, "scalar op benchmark > 0 GFLOPS");
        }
    }
    delete bk;

#if defined(OIL_AVX2)
    BackendConfig cfg2;
    cfg2.type = BackendType::CPU_AVX2;
    ComputeBackend* bk2 = ComputeBackend::create(cfg2);
    if (bk2 && bk2->is_available()) {
        double gflops = benchmark_operation(bk2, "gemm", 256, 256, 256, 5, 30);
        printf("  AVX2 gemm: %.2f GFLOPS\n", gflops);
        CHECK(gflops > 0, "avx2 benchmark > 0 GFLOPS");
    }
    delete bk2;
#endif
}

static void test_auto_select() {
    printf("\n=== AUTO SELECT ===\n");
    HardwareProfile hw = probe_hardware();

    // Small model (100M params @ FP32 ≈ 400 MB)
    BackendConfig small = select_optimal_backend(hw, 400 * 1024 * 1024);
    printf("  Small model (400MB): %s\n", backend_name(small.type));

    // Medium model (1B params @ FP32 ≈ 4 GB)
    BackendConfig medium = select_optimal_backend(hw, (int64_t)4 * 1024 * 1024 * 1024);
    printf("  Medium model (4GB): %s\n", backend_name(medium.type));

    // Large model (13B params @ FP32 ≈ 52 GB)
    BackendConfig large = select_optimal_backend(hw, (int64_t)52LL * 1024 * 1024 * 1024);
    printf("  Large model (52GB): %s\n", backend_name(large.type));

    CHECK(small.type != BackendType::CPU_SCALAR || !probe_hardware().has_avx2,
          "small model gets non-scalar backend when AVX2 available");
    CHECK(medium.type != BackendType::DISTRIBUTED, "medium model gets non-distributed backend");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("MYTHOS.cpp — Benchmark Suite\n");
    printf("============================\n");

    test_hardware_probe();
    test_backend_creation();
    test_bench_gemm();
    test_bench_operations();
    test_auto_select();

    printf("\n==========================\n");
    printf("Results: %d / %d tests passed", tests_passed, tests_total);
    if (tests_passed == tests_total) printf(" -- ALL PASSED\n");
    else printf(" (%d FAILED)\n", tests_total - tests_passed);

    return (tests_passed == tests_total) ? 0 : 1;
}
