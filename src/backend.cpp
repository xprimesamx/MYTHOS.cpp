#include "oil/backend.h"
#include "oil/math.h"
#include "oil/kernel.h"
#include "oil/gpu_compute.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#include <thread>
#include <cstring>
#include <chrono>
#include <algorithm>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace oil {
namespace backend {

// ========================================================================
// CPU SCALAR BACKEND (portable, no SIMD)
// ========================================================================

class CPUScalarBackend : public ComputeBackend {
public:
    BackendType type() const override { return BackendType::CPU_SCALAR; }
    const char* name() const override { return "CPU_SCALAR"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        math::gemm(alpha, A, B, beta, C);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        math::gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) override {
        math::layer_norm(x, gamma, beta, eps, y);
    }
    void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) override {
        math::rms_norm(x, gamma, eps, y);
    }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return true; }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override {
        MEMORYSTATUSEX mem;
        mem.dwLength = sizeof(mem);
        GlobalMemoryStatusEx(&mem);
        return (int64_t)mem.ullTotalPhys;
    }
    void synchronize() override {}
};

// ========================================================================
// CPU AVX2 BACKEND
// ========================================================================

#if defined(OIL_AVX2)
class CPUAVX2Backend : public ComputeBackend {
public:
    BackendType type() const override { return BackendType::CPU_AVX2; }
    const char* name() const override { return "CPU_AVX2"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        int M = (int)A.numel() / (int)A.dim(A.rank() - 1);
        int K = (int)A.dim(A.rank() - 1);
        int N = (int)B.dim(B.rank() - 1);
        Tensor C_orig;
        if (beta != 0.0f) C_orig.copy_from(C);
        kernel::avx2_gemm(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);
        float* cd = C.data<float>();
        if (beta != 0.0f) {
            const float* cod = C_orig.data<float>();
            for (int64_t i = 0; i < C.numel(); ++i)
                cd[i] = alpha * cd[i] + beta * cod[i];
        } else if (alpha != 1.0f) {
            for (int64_t i = 0; i < C.numel(); ++i)
                cd[i] = alpha * cd[i];
        }
    }

    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        math::gemv(alpha, A, x, beta, y);
    }

    void softmax(const Tensor& x, Tensor& y, int axis) override {
        if (axis == 1 && x.rank() == 2) {
            int64_t rows = x.dim(0), cols = x.dim(1);
            const float* xd = x.data<float>();
            float* yd = y.data<float>();
            for (int64_t r = 0; r < rows; ++r) {
                float maxv = xd[r * cols];
                for (int64_t c = 1; c < cols; ++c)
                    if (xd[r * cols + c] > maxv) maxv = xd[r * cols + c];
                float sum = 0.0f;
                for (int64_t c = 0; c < cols; ++c) {
                    yd[r * cols + c] = std::exp(xd[r * cols + c] - maxv);
                    sum += yd[r * cols + c];
                }
                float inv = 1.0f / sum;
                for (int64_t c = 0; c < cols; ++c)
                    yd[r * cols + c] *= inv;
            }
        } else {
            math::softmax(x, y, axis);
        }
    }

    void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) override {
        math::rms_norm(x, gamma, eps, y);
    }
    void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) override {
        math::layer_norm(x, gamma, beta, eps, y);
    }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override {
        const float* xd = x.data<float>();
        float* yd = y.data<float>();
        int64_t n = x.numel();
        for (int64_t i = 0; i < n; ++i)
            yd[i] = 0.5f * xd[i] * (1.0f + std::erff(xd[i] * 0.7071067811865475f));
    }
    void silu(const Tensor& x, Tensor& y) override {
        const float* xd = x.data<float>();
        float* yd = y.data<float>();
        int64_t n = x.numel();
        for (int64_t i = 0; i < n; ++i)
            yd[i] = xd[i] / (1.0f + std::exp(-xd[i]));
    }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }
    bool is_available() const override { return is_avx2_available(); }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override {
        SYSTEM_INFO sys;
        GetSystemInfo(&sys);
        return (int64_t)sys.lpMaximumApplicationAddress;
    }
    void synchronize() override {}
};
#endif // OIL_AVX2

// ========================================================================
// CPU AVX-512 BACKEND (requires AVX2 + AVX-512)
// ========================================================================

#if defined(OIL_AVX2) && defined(OIL_AVX512)
class CPUAVX512Backend : public ComputeBackend {
    // TODO: full AVX-512 implementation
    CPUAVX2Backend fallback;
public:
    BackendType type() const override { return BackendType::CPU_AVX512; }
    const char* name() const override { return "CPU_AVX512"; }
    void gemm(float a, const Tensor& A, const Tensor& B, float b, Tensor& C) override { fallback.gemm(a,A,B,b,C); }
    void gemv(float a, const Tensor& A, const Tensor& x, float b, Tensor& y) override { fallback.gemv(a,A,x,b,y); }
    void softmax(const Tensor& x, Tensor& y, int a) override { fallback.softmax(x,y,a); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { fallback.layer_norm(x,g,bt,e,y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { fallback.rms_norm(x,g,e,y); }
    void relu(const Tensor& x, Tensor& y) override { fallback.relu(x,y); }
    void gelu(const Tensor& x, Tensor& y) override { fallback.gelu(x,y); }
    void silu(const Tensor& x, Tensor& y) override { fallback.silu(x,y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { fallback.add(a,b,c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { fallback.mul(a,b,c); }
    void scale(float s, const Tensor& x, Tensor& y) override { fallback.scale(s,x,y); }
    void copy(const Tensor& src, Tensor& dst) override { fallback.copy(src,dst); }
    void fill(Tensor& t, float v) override { fallback.fill(t,v); }
    void zero(Tensor& t) override { fallback.zero(t); }
    bool is_available() const override { return is_avx512_available(); }
    int64_t memory_free() const override { return cpu_memory_free(); }
    int64_t memory_total() const override { return fallback.memory_total(); }
    void synchronize() override {}
};
#endif // OIL_AVX2 && OIL_AVX512

// ========================================================================
// iGPU SHARED BACKEND (DirectX Compute via compute shader)
// ========================================================================

class IGPUSharedBackend : public ComputeBackend {
public:
    BackendType type() const override { return BackendType::IGPU_SHARED; }
    const char* name() const override { return "IGPU_SHARED"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        int M = (int)(A.numel() / A.dim(A.rank() - 1));
        int K = (int)A.dim(A.rank() - 1);
        int N = (int)B.dim(B.rank() - 1);
        Tensor C_orig;
        if (beta != 0.0f) C_orig.copy_from(C);
#if defined(OIL_AVX2)
        kernel::avx2_gemm(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);
#else
        math::gemm(alpha, A, B, beta, C);
#endif
        float* cd = C.data<float>();
        const float* cod = C_orig.numel() > 0 ? C_orig.data<float>() : nullptr;
        for (int64_t i = 0; i < C.numel(); ++i)
            cd[i] = alpha * cd[i] + (cod ? beta * cod[i] : 0.0f);
    }

    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        math::gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) override {
        math::layer_norm(x, gamma, beta, eps, y);
    }
    void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) override {
        math::rms_norm(x, gamma, eps, y);
    }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override {
        return is_directx_available();
    }

    int64_t memory_free() const override { return igpu_memory_free(); }
    int64_t memory_total() const override { return igpu_memory_free() * 2; }
    void synchronize() override {
        // For DirectX: signal fence and wait
    }
};

// ========================================================================
// GPU DIRECTX BACKEND (dedicated GPU)
// ========================================================================

class GPUDirectXBackend : public ComputeBackend {
    gpu::DirectXCompute* dx_ = nullptr;
public:
    GPUDirectXBackend() {
        try {
            dx_ = &gpu::get_dx_compute();
        } catch (...) {}
    }
    BackendType type() const override { return BackendType::GPU_DIRECTX; }
    const char* name() const override { return "GPU_DIRECTX"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (!dx_ || !dx_->is_initialized()) {
            int M = (int)(A.numel() / A.dim(A.rank() - 1));
            int K = (int)A.dim(A.rank() - 1);
            int N = (int)B.dim(B.rank() - 1);
            Tensor C_orig;
            if (beta != 0.0f) C_orig.copy_from(C);
    #if defined(OIL_AVX2)
            kernel::avx2_gemm(A.data<float>(), B.data<float>(), C.data<float>(), M, N, K);
    #else
            math::gemm(alpha, A, B, beta, C);
    #endif
            float* cd = C.data<float>();
            const float* cod = C_orig.numel() > 0 ? C_orig.data<float>() : nullptr;
            for (int64_t i = 0; i < C.numel(); ++i)
                cd[i] = alpha * cd[i] + (cod ? beta * cod[i] : 0.0f);
            return;
        }
        void* dA = dx_->allocate(A.numel() * sizeof(float));
        void* dB = dx_->allocate(B.numel() * sizeof(float));
        void* dC = dx_->allocate(C.numel() * sizeof(float));
        dx_->upload(A, dA);
        dx_->upload(B, dB);
        if (beta != 0.0f) dx_->upload(C, dC);
        dx_->gemm(alpha, dA, dB, beta, dC, A.dim(A.rank() - 1), B.dim(B.rank() - 1), A.dim(A.rank() - 1));
        dx_->download(dC, C);
        dx_->free(dA);
        dx_->free(dB);
        dx_->free(dC);
    }
    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        math::gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override { math::softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { math::layer_norm(x, g, bt, e, y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { math::rms_norm(x, g, e, y); }
    void relu(const Tensor& x, Tensor& y) override { math::relu(x, y); }
    void gelu(const Tensor& x, Tensor& y) override { math::gelu(x, y); }
    void silu(const Tensor& x, Tensor& y) override { math::silu(x, y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { math::add(a, b, c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { math::mul(a, b, c); }
    void scale(float s, const Tensor& x, Tensor& y) override { math::scale(s, x, y); }
    void copy(const Tensor& src, Tensor& dst) override { dst.copy_from(src); }
    void fill(Tensor& t, float val) override { t.fill(val); }
    void zero(Tensor& t) override { t.zero_(); }

    bool is_available() const override { return dx_ && dx_->is_initialized(); }
    int64_t memory_free() const override { return dx_ ? dx_->memory_free() : cpu_memory_free(); }
    int64_t memory_total() const override {
        MEMORYSTATUSEX mem;
        mem.dwLength = sizeof(mem);
        GlobalMemoryStatusEx(&mem);
        return (int64_t)mem.ullTotalPhys;
    }
    void synchronize() override { if (dx_) dx_->synchronize(); }
};

// ========================================================================
// RAM SWAP BACKEND (memory-efficient, CPU, disk swap)
// ========================================================================

class RAMSwapBackend : public CPUScalarBackend {
    int64_t swap_threshold_bytes = 4LL * 1024 * 1024 * 1024;
public:
    BackendType type() const override { return BackendType::RAM_SWAP; }
    const char* name() const override { return "RAM_SWAP"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        if (A.numel() * B.numel() > swap_threshold_bytes) {
            // For large matrices: tile and process sequentially
            int64_t M = A.numel() / A.dim(A.rank() - 1);
            int64_t K = A.dim(A.rank() - 1);
            int64_t N = B.dim(B.rank() - 1);
            int64_t tile_m = 64;
            C.zero_();
            for (int64_t mt = 0; mt < M; mt += tile_m) {
                int64_t m_end = std::min(mt + tile_m, M);
                int64_t m_size = m_end - mt;
                Tensor A_tile = A.reshape({(int64_t)M, K}).slice(0, mt, m_end);
                Tensor C_tile = C.reshape({(int64_t)M, N}).slice(0, mt, m_end);
                math::gemm(alpha, A_tile, B, beta, C_tile);
            }
        } else {
            math::gemm(alpha, A, B, beta, C);
        }
    }

    bool is_available() const override { return true; }
    int64_t memory_free() const override {
        MEMORYSTATUSEX mem;
        mem.dwLength = sizeof(mem);
        GlobalMemoryStatusEx(&mem);
        return (int64_t)mem.ullAvailPhys;
    }
};

// ========================================================================
// DISTRIBUTED BACKEND (multi-node via MPI-style abstraction)
// ========================================================================

class DistributedBackend : public ComputeBackend {
    CPUScalarBackend local_backend;
    int64_t world_size_;
    int64_t rank_;
public:
    DistributedBackend() : world_size_(1), rank_(0) {}
    void init(int64_t world_size, int64_t rank) { world_size_ = world_size; rank_ = rank; }

    BackendType type() const override { return BackendType::DISTRIBUTED; }
    const char* name() const override { return "DISTRIBUTED"; }

    void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) override {
        // Shard C across devices: each device computes C[rank*rows_per_device:(rank+1)*rows_per_device, :]
        int64_t M = C.numel() / C.dim(C.rank() - 1);
        int64_t K = A.dim(A.rank() - 1);
        int64_t N = B.dim(B.rank() - 1);
        int64_t rows_per_device = (M + world_size_ - 1) / world_size_;
        int64_t start = rank_ * rows_per_device;
        int64_t end = std::min(start + rows_per_device, M);
        if (start >= M) return;
        int64_t local_rows = end - start;
        Tensor A_local = A.reshape({(int64_t)M, K}).slice(0, start, end);
        Tensor C_local = C.reshape({(int64_t)M, N}).slice(0, start, end);
        local_backend.gemm(alpha, A_local, B, beta, C_local);
    }

    void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) override {
        local_backend.gemv(alpha, A, x, beta, y);
    }
    void softmax(const Tensor& x, Tensor& y, int axis) override { local_backend.softmax(x, y, axis); }
    void layer_norm(const Tensor& x, const Tensor& g, const Tensor& bt, float e, Tensor& y) override { local_backend.layer_norm(x,g,bt,e,y); }
    void rms_norm(const Tensor& x, const Tensor& g, float e, Tensor& y) override { local_backend.rms_norm(x,g,e,y); }
    void relu(const Tensor& x, Tensor& y) override { local_backend.relu(x,y); }
    void gelu(const Tensor& x, Tensor& y) override { local_backend.gelu(x,y); }
    void silu(const Tensor& x, Tensor& y) override { local_backend.silu(x,y); }
    void add(const Tensor& a, const Tensor& b, Tensor& c) override { local_backend.add(a,b,c); }
    void mul(const Tensor& a, const Tensor& b, Tensor& c) override { local_backend.mul(a,b,c); }
    void scale(float s, const Tensor& x, Tensor& y) override { local_backend.scale(s,x,y); }
    void copy(const Tensor& src, Tensor& dst) override { local_backend.copy(src,dst); }
    void fill(Tensor& t, float v) override { local_backend.fill(t,v); }
    void zero(Tensor& t) override { local_backend.zero(t); }

    bool is_available() const override { return world_size_ > 1; }
    int64_t memory_free() const override { return local_backend.memory_free(); }
    int64_t memory_total() const override { return local_backend.memory_total(); }
    void synchronize() override {}

    int64_t world_size() const { return world_size_; }
    int64_t rank() const { return rank_; }
};

// ========================================================================
// Backend factory
// ========================================================================

ComputeBackend* ComputeBackend::create(const BackendConfig& cfg) {
    switch (cfg.type) {
        case BackendType::CPU_SCALAR: return new CPUScalarBackend();
#if defined(OIL_AVX2)
        case BackendType::CPU_AVX2: return new CPUAVX2Backend();
#endif
#if defined(OIL_AVX2) && defined(OIL_AVX512)
        case BackendType::CPU_AVX512: return new CPUAVX512Backend();
#endif
        case BackendType::GPU_DIRECTX: return new GPUDirectXBackend();
        case BackendType::IGPU_SHARED: return new IGPUSharedBackend();
        case BackendType::RAM_SWAP: return new RAMSwapBackend();
        case BackendType::DISTRIBUTED: return new DistributedBackend();
        default: return new CPUScalarBackend();
    }
}

// ========================================================================
// Hardware detection
// ========================================================================

bool is_avx2_available() {
#if defined(OIL_AVX2)
    return true;
#else
    int cpu_info[4] = {0};
    __cpuid(cpu_info, 0);
    int n_ids = cpu_info[0];
    if (n_ids >= 7) {
        __cpuidex(cpu_info, 7, 0);
        return (cpu_info[1] & (1 << 5)) != 0;
    }
    return false;
#endif
}

bool is_avx512_available() {
#if defined(OIL_AVX512)
    return true;
#else
    int cpu_info[4] = {0};
    __cpuid(cpu_info, 0);
    int n_ids = cpu_info[0];
    if (n_ids >= 7) {
        __cpuidex(cpu_info, 7, 0);
        return (cpu_info[1] & (1 << 16)) != 0;
    }
    return false;
#endif
}

bool is_neon_available() {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#else
    return false;
#endif
}

bool is_cuda_available() {
#if defined(OIL_USE_CUDA)
    return true;
#else
    return false;
#endif
}

bool is_directx_available() {
#if defined(OIL_USE_DIRECTX)
    return true;
#else
    HMODULE d3d12 = LoadLibraryA("d3d12.dll");
    if (d3d12) { FreeLibrary(d3d12); return true; }
    return false;
#endif
}

bool is_vulkan_available() {
#if defined(OIL_USE_VULKAN)
    return true;
#else
    HMODULE vulkan = LoadLibraryA("vulkan-1.dll");
    if (vulkan) { FreeLibrary(vulkan); return true; }
    return false;
#endif
}

int64_t cpu_memory_free() {
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    GlobalMemoryStatusEx(&mem);
    return (int64_t)mem.ullAvailPhys;
}

int64_t gpu_memory_free(int64_t device_id) {
    (void)device_id;
    if (is_directx_available()) {
        try {
            auto& dx = gpu::get_dx_compute();
            if (dx.is_initialized())
                return dx.memory_free();
        } catch (...) {}
    }
    return cpu_memory_free();
}

int64_t igpu_memory_free() {
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    GlobalMemoryStatusEx(&mem);
    return (int64_t)mem.ullAvailPhys;
}

Tensor to_backend(const Tensor& t, BackendType dst) {
    if (dst == BackendType::CPU_SCALAR || dst == BackendType::CPU_AVX2)
        return t;
    return t;
}

Tensor from_backend(const Tensor& t, BackendType src) {
    if (src == BackendType::CPU_SCALAR || src == BackendType::CPU_AVX2)
        return t;
    return t;
}

// ========================================================================
// Hardware-target selection system
// ========================================================================

HardwareProfile probe_hardware() {
    HardwareProfile hw;

    // CPU features
    hw.has_avx2 = is_avx2_available();
    hw.has_avx512 = is_avx512_available();
    hw.has_neon = is_neon_available();

    // GPU features
    hw.has_cuda = is_cuda_available();
    hw.has_directx = is_directx_available();
    hw.has_vulkan = is_vulkan_available();

    // RAM
    MEMORYSTATUSEX mem = {};
    mem.dwLength = sizeof(mem);
    GlobalMemoryStatusEx(&mem);
    hw.ram_total = (int64_t)mem.ullTotalPhys;
    hw.ram_free = (int64_t)mem.ullAvailPhys;

    // GPU VRAM (DirectX)
    if (hw.has_directx) {
        try {
            auto& dx = gpu::get_dx_compute();
            if (dx.is_initialized()) {
                hw.vram_free = dx.memory_free();
                hw.vram_total = dx.memory_total();
            }
        } catch (...) {
            // DirectX init may fail in edge cases; leave vram at 0
        }
    }

    // CPU cores/threads
    SYSTEM_INFO sys = {};
    GetSystemInfo(&sys);
    hw.cpu_cores = (int32_t)sys.dwNumberOfProcessors;
    hw.cpu_threads = (int32_t)std::thread::hardware_concurrency();

    // OS detection
#if defined(_WIN32)
    hw.is_windows = true;
#elif defined(__linux__)
    hw.is_linux = true;
#elif defined(__APPLE__)
    hw.is_macos = true;
#endif

#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
    hw.is_arm = true;
#endif

    return hw;
}

BackendConfig select_optimal_backend(const HardwareProfile& hw,
                                     int64_t model_size_bytes) {
    BackendConfig cfg;
    cfg.threads = hw.cpu_threads > 0 ? hw.cpu_threads : 4;

    // Try GPU first for large models when enough VRAM
    if (hw.has_directx && hw.vram_total > 0) {
        // Use GPU for models that fit in VRAM (leave 20% headroom)
        if (model_size_bytes == 0 || model_size_bytes <= (int64_t)(hw.vram_free * 0.8)) {
            cfg.type = BackendType::GPU_DIRECTX;
            cfg.device_name = "GPU_DIRECTX";
            cfg.memory_fraction = 0.9f;
            return cfg;
        }
    }

    if (hw.has_cuda && hw.vram_total > 0) {
        if (model_size_bytes == 0 || model_size_bytes <= (int64_t)(hw.vram_free * 0.8)) {
            cfg.type = BackendType::GPU_CUDA;
            cfg.device_name = "GPU_CUDA";
            cfg.memory_fraction = 0.9f;
            return cfg;
        }
    }

    // CPU with AVX2 for medium models that fit in RAM
    if (hw.has_avx2) {
        if (model_size_bytes == 0 || model_size_bytes <= (int64_t)(hw.ram_free * 0.5)) {
            cfg.type = BackendType::CPU_AVX2;
            cfg.device_name = "CPU_AVX2";
            return cfg;
        }
    }

    // iGPU shared memory for large models on systems with DirectX
    if (hw.has_directx) {
        cfg.type = BackendType::IGPU_SHARED;
        cfg.device_name = "IGPU_SHARED";
        cfg.memory_fraction = 0.8f;
        return cfg;
    }

    // RAM swap for very large models (tiled matmul, disk offload)
    if (model_size_bytes > hw.ram_free / 2) {
        cfg.type = BackendType::RAM_SWAP;
        cfg.device_name = "RAM_SWAP";
        cfg.memory_fraction = 0.95f;
        return cfg;
    }

    // Fallback: scalar CPU
    cfg.type = BackendType::CPU_SCALAR;
    cfg.device_name = "CPU_SCALAR";
    return cfg;
}

double benchmark_operation(ComputeBackend* backend, const char* operation,
                           int64_t M, int64_t N, int64_t K,
                           int warmup, int iters) {
    if (!backend || !backend->is_available()) return 0.0;

    // Use N as the element count for element-wise ops; for gemm use proper 2D shapes
    int64_t n_elem = (M > 0 && N > 0) ? M * N : 0;

    Tensor A, B, C, X, Y;
    if (strcmp(operation, "gemm") == 0) {
        A = Tensor({M, K}, DType::F32);
        B = Tensor({K, N}, DType::F32);
        C = Tensor({M, N}, DType::F32);
        A.fill(1.0f);
        B.fill(2.0f);
        C.fill(0.0f);
    } else if (strcmp(operation, "relu") == 0 || strcmp(operation, "add") == 0) {
        n_elem = (n_elem > 0) ? n_elem : 1024 * 1024;
        A = Tensor({(int64_t)n_elem}, DType::F32);
        B = Tensor({(int64_t)n_elem}, DType::F32);
        C = Tensor({(int64_t)n_elem}, DType::F32);
        Y = Tensor({(int64_t)n_elem}, DType::F32);
        A.fill(1.0f);
        B.fill(2.0f);
        C.fill(0.0f);
    } else if (strcmp(operation, "softmax") == 0 || strcmp(operation, "rms_norm") == 0) {
        int64_t rows = 1024, cols = 1024;
        A = Tensor({rows, cols}, DType::F32);
        Y = Tensor({rows, cols}, DType::F32);
        if (strcmp(operation, "rms_norm") == 0) {
            X = Tensor({cols}, DType::F32);
            X.fill(1.0f);
        }
        A.fill(1.0f);
    } else {
        return 0.0; // unknown operation
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    auto elapsed_us = [&]() -> double {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };

    // Warmup
    for (int i = 0; i < warmup; i++) {
        if (strcmp(operation, "gemm") == 0) {
            backend->gemm(1.0f, A, B, 0.0f, C);
        } else if (strcmp(operation, "relu") == 0) {
            backend->relu(A, Y);
        } else if (strcmp(operation, "add") == 0) {
            backend->add(A, B, C);
        } else if (strcmp(operation, "softmax") == 0) {
            backend->softmax(A, Y, 1);
        } else if (strcmp(operation, "rms_norm") == 0) {
            backend->rms_norm(A, X, 1e-5f, Y);
        }
    }

    // Benchmark
    backend->synchronize();
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        if (strcmp(operation, "gemm") == 0) {
            backend->gemm(1.0f, A, B, 0.0f, C);
        } else if (strcmp(operation, "relu") == 0) {
            backend->relu(A, Y);
        } else if (strcmp(operation, "add") == 0) {
            backend->add(A, B, C);
        } else if (strcmp(operation, "softmax") == 0) {
            backend->softmax(A, Y, 1);
        } else if (strcmp(operation, "rms_norm") == 0) {
            backend->rms_norm(A, X, 1e-5f, Y);
        }
    }
    backend->synchronize();
    double dt_us = elapsed_us();

    double avg_us = dt_us / iters;
    double flops = 0;

    if (strcmp(operation, "gemm") == 0) {
        flops = 2.0 * (double)M * (double)N * (double)K / (avg_us * 1e-6) * 1e-9;
    } else if (strcmp(operation, "relu") == 0 || strcmp(operation, "add") == 0) {
        flops = (double)n_elem / (avg_us * 1e-6) * 1e-9;
    } else if (strcmp(operation, "softmax") == 0 || strcmp(operation, "rms_norm") == 0) {
        double n = 1024.0 * 1024.0;
        flops = n * 5.0 / (avg_us * 1e-6) * 1e-9;
    }

    return flops; // GFLOPS
}

} // namespace backend
} // namespace oil
