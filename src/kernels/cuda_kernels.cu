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
// New kernels: RoPE, masked softmax, cross-entropy, AdamW, dropout,
// SwiGLU, fused RMSNorm+add, gradient clip
// ========================================================================

// E10: CUDA fused RoPE — applies rotary position embedding in-place
__global__ void cuda_rope_kernel(float* q, float* k,
                                  const float* cos_cache, const float* sin_cache,
                                  int B, int H, int S, int D, int seq_start) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_d = D / 2;
    int total = B * H * S * half_d;
    if (idx >= total) return;

    int d = idx % half_d;
    int s = (idx / half_d) % S;
    int h = (idx / (half_d * S)) % H;
    int b = idx / (half_d * S * H);

    int pos = seq_start + s;
    float c = cos_cache[pos * half_d + d];
    float sn = sin_cache[pos * half_d + d];

    float q0 = q[((b * H + h) * S + s) * D + d];
    float q1 = q[((b * H + h) * S + s) * D + d + half_d];
    q[((b * H + h) * S + s) * D + d] = q0 * c - q1 * sn;
    q[((b * H + h) * S + s) * D + d + half_d] = q0 * sn + q1 * c;

    float k0 = k[((b * H + h) * S + s) * D + d];
    float k1 = k[((b * H + h) * S + s) * D + d + half_d];
    k[((b * H + h) * S + s) * D + d] = k0 * c - k1 * sn;
    k[((b * H + h) * S + s) * D + d + half_d] = k0 * sn + k1 * c;
}

// E11: CUDA fused attention scores — Q @ K^T * scale with optional causal mask
__global__ void cuda_attn_scores_kernel(float* score, const float* Q, const float* K,
                                         float scale, int B, int H, int S, int S_full,
                                         int D, int kv_h, bool apply_causal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S * S_full;
    if (idx >= total) return;

    int t = idx % S_full;
    int s = (idx / S_full) % S;
    int h = (idx / (S_full * S)) % H;
    int b = idx / (S_full * S * H);

    if (apply_causal && t > s) {
        score[idx] = -INFINITY;
        return;
    }

    int kh = h % kv_h;
    float sum = 0;
    for (int d = 0; d < D; d++) {
        sum += Q[((b * H + h) * S + s) * D + d]
             * K[(kh * S_full + t) * D + d];
    }
    score[idx] = sum * scale;
}

// E12: CUDA masked softmax — with fused causal mask support
__global__ void cuda_masked_softmax_kernel(float* y, const float* x,
                                            const float* mask, int rows, int cols,
                                            int mask_stride) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    float max_val = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = row_x[c];
        if (mask) v += mask[row / (cols / mask_stride) * mask_stride + c % mask_stride];
        shared[threadIdx.x] = fmaxf(shared[threadIdx.x], v);
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = row_x[c];
        if (mask) v += mask[row / (cols / mask_stride) * mask_stride + c % mask_stride];
        if (v > local_max) local_max = v;
    }
    shared[threadIdx.x] = local_max;
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
        float v = row_x[c];
        if (mask) v += mask[row / (cols / mask_stride) * mask_stride + c % mask_stride];
        float e = expf(v - max_val);
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

// E14: CUDA cross-entropy loss
__global__ void cuda_cross_entropy_kernel(float* loss, const float* logits,
                                           const int* targets, int B, int V) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const float* row = logits + b * V;
    float max_val = row[0];
    for (int v = 1; v < V; v++)
        if (row[v] > max_val) max_val = row[v];

    float sum_exp = 0;
    for (int v = 0; v < V; v++)
        sum_exp += expf(row[v] - max_val);

    int t = targets[b];
    loss[b] = -(row[t] - max_val - logf(sum_exp));
}

// E15: CUDA cross-entropy gradient
__global__ void cuda_cross_entropy_grad_kernel(float* grad, const float* logits,
                                                const int* targets,
                                                float* loss_scale,
                                                int B, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * V;
    if (idx >= total) return;

    int v = idx % V;
    int b = idx / V;

    float max_val = -INFINITY;
    for (int vv = 0; vv < V; vv++)
        if (logits[b * V + vv] > max_val) max_val = logits[b * V + vv];

    float sum_exp = 0;
    for (int vv = 0; vv < V; vv++)
        sum_exp += expf(logits[b * V + vv] - max_val);

    float prob = expf(logits[idx] - max_val) / sum_exp;
    float scale = (v == targets[b]) ? (prob - 1.0f) : prob;
    grad[idx] = scale * (*loss_scale) / (float)B;
}

// E16: CUDA AdamW optimizer step
__global__ void cuda_adamw_kernel(float* param, float* grad, float* m, float* v,
                                   float beta1, float beta2, float eps, float lr,
                                   float weight_decay, int64_t t, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    g += weight_decay * param[i];

    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

    float m_hat = m[i] / (1.0f - powf(beta1, (float)t));
    float v_hat = v[i] / (1.0f - powf(beta2, (float)t));

    param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// E17: CUDA dropout
__global__ void cuda_dropout_kernel(float* out, const float* x, const float* mask,
                                     float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * mask[i] * scale;
}

// E18: CUDA fused SwiGLU — output = silu(gate) * up
__global__ void cuda_swiglu_kernel(float* out, const float* gate,
                                    const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float s = g / (1.0f + expf(-g));
        out[i] = s * up[i];
    }
}

// E19: CUDA fused RMSNorm + residual add
__global__ void cuda_rmsnorm_add_kernel(float* out, const float* x,
                                          const float* residual,
                                          const float* gamma,
                                          float eps, int n, int d) {
    int row = blockIdx.x;
    if (row >= n) return;
    extern __shared__ float shared[];
    const float* row_x = x + row * d;
    float* row_out = out + row * d;

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
        row_out[i] = row_x[i] * inv_rms * gamma[i] + residual[row * d + i];
}

// E20: CUDA gradient clip by norm
__global__ void cuda_clip_grad_kernel(float* grad, float max_norm,
                                       float* grad_norm_sq, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_norm_sq[i] = grad[i] * grad[i];
}

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

void launch_cuda_moe_gather(float* out, const float* x,
                             const int64_t* indices, const float* weights,
                             int T, int K, int D) {
    int total = T * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cuda_moe_gather_kernel<<<blocks, threads>>>(out, x, indices, weights, T, K, D);
}

void launch_cuda_rope(float* q, float* k, const float* cos_cache,
                       const float* sin_cache, int B, int H, int S, int D,
                       int seq_start) {
    int half_d = D / 2;
    int total = B * H * S * half_d;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cuda_rope_kernel<<<blocks, threads>>>(q, k, cos_cache, sin_cache, B, H, S, D, seq_start);
}

void launch_cuda_attn_scores(float* score, const float* Q, const float* K,
                              float scale, int B, int H, int S, int S_full,
                              int D, int kv_h, bool apply_causal) {
    int total = B * H * S * S_full;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cuda_attn_scores_kernel<<<blocks, threads>>>(score, Q, K, scale, B, H, S, S_full, D, kv_h, apply_causal);
}

void launch_cuda_masked_softmax(float* y, const float* x, const float* mask,
                                 int rows, int cols, int mask_stride) {
    int threads = min(cols, 256);
    int blocks = rows;
    cuda_masked_softmax_kernel<<<blocks, threads, threads * sizeof(float)>>>(y, x, mask, rows, cols, mask_stride);
}

void launch_cuda_cross_entropy(float* loss, const float* logits,
                                const int* targets, int B, int V) {
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    cuda_cross_entropy_kernel<<<blocks, threads>>>(loss, logits, targets, B, V);
}

void launch_cuda_cross_entropy_grad(float* grad, const float* logits,
                                     const int* targets, float* loss_scale,
                                     int B, int V) {
    int total = B * V;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cuda_cross_entropy_grad_kernel<<<blocks, threads>>>(grad, logits, targets, loss_scale, B, V);
}

void launch_cuda_adamw(float* param, float* grad, float* m, float* v,
                        float beta1, float beta2, float eps, float lr,
                        float weight_decay, int64_t t, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cuda_adamw_kernel<<<blocks, threads>>>(param, grad, m, v, beta1, beta2, eps, lr, weight_decay, t, n);
}

void launch_cuda_dropout(float* out, const float* x, const float* mask,
                          float scale, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cuda_dropout_kernel<<<blocks, threads>>>(out, x, mask, scale, n);
}

void launch_cuda_swiglu(float* out, const float* gate, const float* up, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cuda_swiglu_kernel<<<blocks, threads>>>(out, gate, up, n);
}

void launch_cuda_rmsnorm_add(float* out, const float* x, const float* residual,
                               const float* gamma, float eps, int n, int d) {
    int threads = min(d, 256);
    int blocks = n;
    cuda_rmsnorm_add_kernel<<<blocks, threads, threads * sizeof(float)>>>(out, x, residual, gamma, eps, n, d);
}

void launch_cuda_clip_grad(float* grad, float max_norm, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    float* d_norm_sq;
    cudaMalloc(&d_norm_sq, sizeof(float));
    cuda_clip_grad_kernel<<<blocks, threads>>>(grad, max_norm, d_norm_sq, n);
    float h_norm_sq = 0;
    cudaMemcpy(&h_norm_sq, d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost);
    float norm = sqrtf(h_norm_sq);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        launch_cuda_scale(grad, grad, scale, n);
    }
    cudaFree(d_norm_sq);
}

void launch_cuda_gemm(float alpha, const float* A, const float* B,
                       float beta, float* C, int M, int N, int K) {
    // Host-side GEMM fallback (cuBLAS used in CUDABackend::gemm)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += A[m * K + k] * B[k * N + n];
            C[m * N + n] = alpha * sum + beta * C[m * N + n];
        }
    }
}

} // extern "C"

#endif // __CUDACC__
