#pragma once
#include "oil/tensor.h"
#include <vector>
#include <cstdint>

namespace oil {
namespace gpu {

enum class GPUType {
    DIRECTX12,
    CUDA,
    VULKAN
};

struct GPUConfig {
    GPUType type = GPUType::DIRECTX12;
    int64_t device_id = 0;
    bool enable_fp16 = true;
    bool enable_tf32 = true;
    int64_t num_warps = 32;
    int64_t workgroup_size = 256;
};

class DirectXCompute {
public:
    DirectXCompute();
    ~DirectXCompute();

    bool init(int64_t device_id = 0);
    bool is_initialized() const;
    void shutdown();

    // Memory management
    void* allocate(size_t bytes);
    void free(void* ptr);
    void upload(const Tensor& src, void* dst);
    void download(void* src, Tensor& dst);
    void copy(void* dst, const void* src, size_t bytes);

    // GEMM: C = alpha * A * B + beta * C
    void gemm(float alpha, const void* A, const void* B, float beta, void* C,
              int64_t M, int64_t N, int64_t K);

    // GEMV: y = alpha * A * x + beta * y
    void gemv(float alpha, const void* A, const void* x, float beta, void* y,
              int64_t M, int64_t N);

    // Element-wise operations
    void relu(const void* x, void* y, int64_t n);
    void gelu(const void* x, void* y, int64_t n);
    void silu(const void* x, void* y, int64_t n);
    void add(const void* a, const void* b, void* c, int64_t n);
    void mul(const void* a, const void* b, void* c, int64_t n);
    void scale(float s, const void* x, void* y, int64_t n);

    // Normalization
    void softmax(const void* x, void* y, int64_t rows, int64_t cols);
    void rms_norm(const void* x, const void* gamma, void* y, float eps, int64_t n, int64_t d);
    void layer_norm(const void* x, const void* gamma, const void* beta, void* y, float eps, int64_t n, int64_t d);

    // MoE kernels
    void moe_gather(const void* x, const int64_t* indices, const float* weights,
                    void* out, int64_t T, int64_t K, int64_t D);
    void moe_scatter_add(void* out, const int64_t* indices, const float* weights,
                         const void* expert_out, int64_t T, int64_t K, int64_t D);

    // Synchronize
    void synchronize();
    int64_t memory_free() const;
    int64_t memory_total() const;

private:
    struct Impl;
    Impl* impl_;
};

// ========================================================================
// CUDA backend (when CUDA is available)
// ========================================================================

class CUDABackend {
public:
    CUDABackend();
    ~CUDABackend();

    bool init(int64_t device_id = 0);
    bool is_initialized() const;
    void shutdown();

    void* allocate(size_t bytes);
    void free(void* ptr);
    void upload(const Tensor& src, void* dst);
    void download(void* src, Tensor& dst);

    void gemm(float alpha, const void* A, const void* B, float beta, void* C,
              int64_t M, int64_t N, int64_t K);
    void gemv(float alpha, const void* A, const void* x, float beta, void* y,
              int64_t M, int64_t N);

    void relu(const void* x, void* y, int64_t n);
    void gelu(const void* x, void* y, int64_t n);
    void silu(const void* x, void* y, int64_t n);
    void add(const void* a, const void* b, void* c, int64_t n);
    void mul(const void* a, const void* b, void* c, int64_t n);
    void scale(float s, const void* x, void* y, int64_t n);

    void softmax(const void* x, void* y, int64_t rows, int64_t cols);
    void rms_norm(const void* x, const void* gamma, void* y, float eps,
                  int64_t n, int64_t d);
    void layer_norm(const void* x, const void* gamma, const void* beta, void* y,
                    float eps, int64_t n, int64_t d);

    void moe_gather(const void* x, const int64_t* indices, const float* weights,
                    void* out, int64_t T, int64_t K, int64_t D);
    void moe_scatter_add(void* out, const int64_t* indices, const float* weights,
                         const void* expert_out, int64_t T, int64_t K, int64_t D);

    void synchronize();
    int64_t memory_free() const;
    int64_t memory_total() const;

private:
    struct Impl;
    Impl* impl_;
};

// ========================================================================
// GPU autodetection and factory
// ========================================================================

GPUType detect_best_gpu();
DirectXCompute& get_dx_compute();
CUDABackend& get_cuda_backend();
bool gpu_available();

void init_gpu(GPUType type = GPUType::DIRECTX12, int64_t device = 0);
void shutdown_gpu();

} // namespace gpu
} // namespace oil
