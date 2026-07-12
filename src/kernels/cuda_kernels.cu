#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// E2: CUDA softmax — block-wide online softmax
__global__ void cuda_softmax_kernel(float* y, const float* x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    float max_val = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        max_val = fmaxf(max_val, row_x[c]);
    shared[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = shared[0];
    __syncthreads();

    float sum = 0;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float e = expf(row_x[c] - max_val);
        row_y[c] = e;
        sum += e;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    sum = shared[0];
    __syncthreads();

    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        row_y[c] /= sum;
}

// E3: CUDA layer_norm
__global__ void cuda_layernorm_kernel(float* y, const float* x,
                                       const float* gamma, const float* beta,
                                       float eps, int n, int d) {
    int row = blockIdx.x;
    if (row >= n) return;
    extern __shared__ float shared[];
    const float* row_x = x + row * d;
    float* row_y = y + row * d;

    float sum = 0, sq_sum = 0;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        sum += row_x[i];
        sq_sum += row_x[i] * row_x[i];
    }
    shared[threadIdx.x] = sum;
    shared[blockDim.x + threadIdx.x] = sq_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
            shared[blockDim.x + threadIdx.x] += shared[blockDim.x + threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean = shared[0] / d;
    float var = shared[blockDim.x] / d - mean * mean;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < d; i += blockDim.x)
        row_y[i] = (row_x[i] - mean) * inv_std * gamma[i] + (beta ? beta[i] : 0.0f);
}

// E4: CUDA rms_norm
__global__ void cuda_rmsnorm_kernel(float* y, const float* x,
                                     const float* gamma, float eps, int n, int d) {
    int row = blockIdx.x;
    if (row >= n) return;
    extern __shared__ float shared[];
    const float* row_x = x + row * d;
    float* row_y = y + row * d;

    float sq_sum = 0;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        sq_sum += row_x[i] * row_x[i];
    shared[threadIdx.x] = sq_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }

    float rms = sqrtf(shared[0] / d + eps);
    float inv_rms = 1.0f / rms;
    __syncthreads();

    for (int i = threadIdx.x; i < d; i += blockDim.x)
        row_y[i] = row_x[i] * inv_rms * gamma[i];
}

// E6: CUDA element-wise kernels
__global__ void cuda_relu_kernel(float* y, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaxf(0.0f, x[i]);
}

__global__ void cuda_gelu_kernel(float* y, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float c = xi * (0.035677f * xi * xi + 0.797885f);
        y[i] = 0.5f * xi * (1.0f + tanhf(c));
    }
}

__global__ void cuda_silu_kernel(float* y, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] / (1.0f + expf(-x[i]));
}

__global__ void cuda_add_kernel(float* c, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void cuda_mul_kernel(float* c, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

__global__ void cuda_scale_kernel(float* y, const float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * s;
}

// E7: CUDA embedding lookup
__global__ void cuda_embedding_kernel(float* out, const float* table,
                                       const int* indices, int B, int S, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * D;
    if (idx >= total) return;
    int d = idx % D;
    int s = (idx / D) % S;
    int b = idx / (S * D);
    int token = indices[b * S + s];
    out[idx] = table[token * D + d];
}

// E8: CUDA MoE dispatch
__global__ void cuda_moe_gather_kernel(float* out, const float* x,
                                        const int64_t* indices, const float* weights,
                                        int T, int K, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * D) return;
    int d = idx % D;
    int t = idx / D;
    float val = 0;
    for (int k = 0; k < K; k++) {
        int expert = (int)indices[t * K + k];
        float w = weights[t * K + k];
        val += w * x[expert * T * D + t * D + d];
    }
    out[t * D + d] = val;
}

// E9: CUDA memory pool — managed via host-side free list
// (Kernel memory management is host-side, not in .cu file)

// ========================================================================
// Launch wrappers (extern "C" for C++ linkage from gpu_compute.cpp)
// ========================================================================

extern "C" {

void launch_cuda_softmax(float* y, const float* x, int rows, int cols) {
    int threads = min(cols, 256);
    int blocks = rows;
    cuda_softmax_kernel<<<blocks, threads, threads * sizeof(float)>>>(y, x, rows, cols);
}

void launch_cuda_layernorm(float* y, const float* x, const float* gamma,
                            const float* beta, float eps, int n, int d) {
    int threads = min(d, 256);
    cuda_layernorm_kernel<<<n, threads, 2 * threads * sizeof(float)>>>(y, x, gamma, beta, eps, n, d);
}

void launch_cuda_rmsnorm(float* y, const float* x, const float* gamma,
                          float eps, int n, int d) {
    int threads = min(d, 256);
    cuda_rmsnorm_kernel<<<n, threads, threads * sizeof(float)>>>(y, x, gamma, eps, n, d);
}

void launch_cuda_relu(float* y, const float* x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cuda_relu_kernel<<<blocks, threads>>>(y, x, n);
}

void launch_cuda_gelu(float* y, const float* x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cuda_gelu_kernel<<<blocks, threads>>>(y, x, n);
}

void launch_cuda_silu(float* y, const float* x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cuda_silu_kernel<<<blocks, threads>>>(y, x, n);
}

void launch_cuda_add(float* c, const float* a, const float* b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cuda_add_kernel<<<blocks, threads>>>(c, a, b, n);
}

void launch_cuda_mul(float* c, const float* a, const float* b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cuda_mul_kernel<<<blocks, threads>>>(c, a, b, n);
}

void launch_cuda_scale(float* y, const float* x, float s, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cuda_scale_kernel<<<blocks, threads>>>(y, x, s, n);
}

void launch_cuda_embedding(float* out, const float* table,
                            const int* indices, int B, int S, int D) {
    int total = B * S * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cuda_embedding_kernel<<<blocks, threads>>>(out, table, indices, B, S, D);
}

} // extern "C"

#endif // __CUDACC__
