#include "oil/autograd.h"
#include "oil/math.h"
#include "oil/kernel.h"
#include <cmath>
#include <cstring>
#include <cfloat>

#include <unordered_set>
#include <cstring>

namespace oil {

bool AutogradEngine::enabled_ = false;

// Forward declarations for loss functions used by CrossEntropyFunction
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets);
Tensor cross_entropy_grad(const Tensor& logits, const Tensor& targets);

// ========================================================================
// REAL matmul_grad: dA = grad_output * B^T (using scalar_gemm convention)
// ========================================================================

Tensor matmul_grad(const Tensor& a, const Tensor& b, const Tensor& grad_output) {
    return matmul_grad_wrt_a(grad_output, b);
}

// dA = grad_output * B^T where B is stored as {N, K} (weight shape)
// grad_output: {M, N},  b (weight): {N, K}
// dA: {M, K}
// Access pattern matches scalar_gemm: bd[k * N + j] where k ∈ [0,K), j ∈ [0,N)
Tensor matmul_grad_wrt_a(const Tensor& grad_output, const Tensor& b) {
    int64_t M = grad_output.dim(0);
    int64_t N = b.dim(0);
    int64_t K = b.dim(1);
    Tensor dA({M, K});
    const float* gd = grad_output.data<float>();
    const float* bd = b.data<float>();
    float* dad = dA.data<float>();
    for (int64_t i = 0; i < M; ++i)
        for (int64_t k = 0; k < K; ++k) {
            float s = 0.0f;
            for (int64_t j = 0; j < N; ++j)
                s += gd[i * N + j] * bd[k * N + j];
            dad[i * K + k] = s;
        }
    return dA;
}

// dB = A^T * grad_output  where B is stored as {N, K}
// grad_output: {M, N},  a: {M, K}
// Returns: gradient in {K, N} format (matching scalar_gemm access pattern)
Tensor matmul_grad_wrt_b(const Tensor& grad_output, const Tensor& a) {
    int64_t K = a.dim(1);
    int64_t M = a.dim(0);
    int64_t N = grad_output.dim(1);
    Tensor dB({K, N});
    const float* gd = grad_output.data<float>();
    const float* ad = a.data<float>();
    float* dbd = dB.data<float>();
    for (int64_t k = 0; k < K; ++k)
        for (int64_t j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int64_t i = 0; i < M; ++i)
                s += ad[i * K + k] * gd[i * N + j];
            dbd[k * N + j] = s;
        }
    return dB;
}

// Gradient w.r.t. weight stored as {N, K} (transpose of matmul_grad_wrt_b output)
static Tensor weight_grad(const Tensor& grad_output, const Tensor& a, const Tensor& weight) {
    // weight: {N, K}, a: {M, K}, grad_output: {M, N}
    // dB from matmul_grad_wrt_b is {K, N} with dB[k*N + n] = gradient for weight_flat[k*N + n]
    // But weight is stored as {N, K}, so we need to transpose the gradient data
    int64_t N = weight.dim(0);
    int64_t K = weight.dim(1);
    Tensor dB = matmul_grad_wrt_b(grad_output, a);  // {K, N}
    // Reinterpret: the flat gradient at offset k*N + n in dB corresponds to
    // weight gradient at offset k*N + n (flat), but weight stores at n*K + k
    // So we need to transpose: result[n*K + k] = dB[k*N + n]
    Tensor dW({N, K}, DType::F32);
    const float* dbd = dB.data<float>();
    float* dwd = dW.data<float>();
    for (int64_t k = 0; k < K; ++k)
        for (int64_t n = 0; n < N; ++n)
            dwd[n * K + k] = dbd[k * N + n];
    return dW;
}

// ========================================================================
// Autograd Function classes
// ========================================================================

class MatMulFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        const Tensor& a = inputs[0];
        const Tensor& b = inputs[1];
        int64_t M = a.dim(0);
        int64_t K = a.dim(1);
        int64_t N = b.dim(0);
        Tensor out({M, N}, DType::F32);
        kernel::scalar_gemm(
            a.data<float>(), b.data<float>(),
            out.data<float>(), (int)M, (int)N, (int)K
        );
        return {out};
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        const Tensor& g_orig = grad_output[0];
        const Tensor& a = saved[0];
        const Tensor& b = saved[1];
        // Flatten gradient to 2D {M, N} if higher-rank (to match forward output shape)
        int64_t M = a.dim(0);
        int64_t K = a.dim(1); (void)K;
        int64_t N = b.dim(0);
        Tensor g = g_orig;
        if (g_orig.rank() > 2) {
            g = g_orig.reshape(Shape{M, N});
        }
        Tensor ga = matmul_grad_wrt_a(g, b);
        Tensor gb = weight_grad(g, a, b);
        return {ga, gb};
    }
};

class AddFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        Tensor out(inputs[0].shape(), DType::F32);
        math::add(inputs[0], inputs[1], out);
        return {out};
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        // Gradient flows equally to both inputs
        return {grad_output[0], grad_output[0]};
    }
};

class SiLUFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        Tensor out(inputs[0].shape(), DType::F32);
        math::silu(inputs[0], out);
        return {out};
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        Tensor g = silu_grad(saved[0], grad_output[0]);
        return {g};
    }
};

class MulFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        Tensor out(inputs[0].shape(), DType::F32);
        math::mul(inputs[0], inputs[1], out);
        return {out};
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        const Tensor& g = grad_output[0];
        const Tensor& a = saved[0];
        const Tensor& b = saved[1];
        // d(a*b)/da = b,  d(a*b)/db = a
        Tensor ga(a.shape(), DType::F32);
        Tensor gb(b.shape(), DType::F32);
        math::mul(g, b, ga);
        math::mul(g, a, gb);
        return {ga, gb};
    }
};

class RMSNormFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs; // inputs[0]=x, inputs[1]=gamma
        Tensor out(inputs[0].shape(), DType::F32);
        float eps = saved_eps;
        math::rms_norm(inputs[0], inputs[1], eps, out);
        return {out};
    }

    float saved_eps = 1e-5f;

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        const Tensor& x = saved[0];
        const Tensor& gamma = saved[1];
        int64_t N = x.numel() / x.dim(x.rank() - 1);
        Tensor dgamma({gamma.numel()}, DType::F32);
        dgamma.zero_();
        Tensor dx = rms_norm_grad(x, gamma, grad_output[0], (int)N, &dgamma);
        return {dx, dgamma};
    }
};

class CrossEntropyFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        Tensor loss = cross_entropy_loss(inputs[0], inputs[1]);
        return {loss};
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        Tensor grad = cross_entropy_grad(saved[0], saved[1]);
        if (grad_output[0].numel() == 1) {
            float scale = grad_output[0].data<float>()[0];
            math::scale(scale, grad, grad);
        }
        return {grad};
    }
};

// ========================================================================
// RotaryFunction: applies RoPE rotation (differentiable)
// ========================================================================

class RotaryFunction : public AutogradFunction {
public:
    RotaryFunction(int64_t hd, const Tensor& cos, const Tensor& sin,
                   int64_t ss, int64_t sl)
        : head_dim_(hd), cos_cached_(cos), sin_cached_(sin),
          seq_start_(ss), seq_len_(sl), B_(0), H_(0), S_(0), D_(0) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        const Tensor& x = inputs[0];
        B_ = x.dim(0); H_ = x.dim(1); S_ = x.dim(2); D_ = x.dim(3);
        Tensor out(x.shape(), DType::F32);
        const float* xd = x.data<float>();
        float* od = out.data<float>();
        const float* cos_d = cos_cached_.data<float>();
        const float* sin_d = sin_cached_.data<float>();
        int64_t half = D_ / 2;
        for (int64_t b = 0; b < B_; b++)
            for (int64_t h = 0; h < H_; h++)
                for (int64_t s = 0; s < S_; s++) {
                    int64_t base = ((b * H_ + h) * S_ + s) * D_;
                    int64_t pos = seq_start_ + s;
                    for (int64_t d = 0; d < half; d++) {
                        float x1 = xd[base + d];
                        float x2 = xd[base + d + half];
                        float cos_v = cos_d[pos * half + d];
                        float sin_v = sin_d[pos * half + d];
                        od[base + d] = x1 * cos_v - x2 * sin_v;
                        od[base + d + half] = x1 * sin_v + x2 * cos_v;
                    }
                }
        return {out};
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        const Tensor& grad = grad_output[0];
        Tensor dx(grad.shape(), DType::F32);
        const float* gd = grad.data<float>();
        float* dxd = dx.data<float>();
        const float* cos_d = cos_cached_.data<float>();
        const float* sin_d = sin_cached_.data<float>();
        int64_t half = D_ / 2;
        for (int64_t b = 0; b < B_; b++)
            for (int64_t h = 0; h < H_; h++)
                for (int64_t s = 0; s < S_; s++) {
                    int64_t base = ((b * H_ + h) * S_ + s) * D_;
                    int64_t pos = seq_start_ + s;
                    for (int64_t d = 0; d < half; d++) {
                        float g1 = gd[base + d];
                        float g2 = gd[base + d + half];
                        float cos_v = cos_d[pos * half + d];
                        float sin_v = sin_d[pos * half + d];
                        dxd[base + d] = g1 * cos_v + g2 * sin_v;
                        dxd[base + d + half] = -g1 * sin_v + g2 * cos_v;
                    }
                }
        return {dx};
    }

private:
    int64_t head_dim_;
    Tensor cos_cached_, sin_cached_;
    int64_t seq_start_, seq_len_;
    int64_t B_, H_, S_, D_;
};

#if defined(OIL_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

static void softmax_forward_avx2(float* output, const float* input, int64_t cols) {
    int64_t i;
    float max_val = -FLT_MAX;
    for (i = 0; i < cols; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum_exp = 0;
#if defined(OIL_AVX2) || defined(__AVX2__)
    __m256 sumv = _mm256_setzero_ps();
    __m256 max_bc = _mm256_set1_ps(max_val);
    __m256 clamp_lo = _mm256_set1_ps(-20.0f);
    __m256 zero8 = _mm256_setzero_ps();
    for (i = 0; i + 8 <= cols; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        v = _mm256_sub_ps(v, max_bc);
        v = _mm256_max_ps(v, clamp_lo);
        __m256 x = v;
        __m256 result = _mm256_set1_ps(1.0f);
        __m256 term = x;
        result = _mm256_add_ps(result, term);
        term = _mm256_mul_ps(term, x);
        result = _mm256_add_ps(result, _mm256_mul_ps(term, _mm256_set1_ps(1.0f/2.0f)));
        term = _mm256_mul_ps(term, x);
        result = _mm256_add_ps(result, _mm256_mul_ps(term, _mm256_set1_ps(1.0f/6.0f)));
        term = _mm256_mul_ps(term, x);
        result = _mm256_add_ps(result, _mm256_mul_ps(term, _mm256_set1_ps(1.0f/24.0f)));
        result = _mm256_blendv_ps(result, zero8, _mm256_cmp_ps(v, clamp_lo, _CMP_LE_OQ));
        _mm256_storeu_ps(output + i, result);
        sumv = _mm256_add_ps(sumv, result);
    }
    float hsum[8];
    _mm256_storeu_ps(hsum, sumv);
    sum_exp = hsum[0] + hsum[1] + hsum[2] + hsum[3] + hsum[4] + hsum[5] + hsum[6] + hsum[7];
#else
    for (i = 0; i < cols; i++) {
        float e = expf(input[i] - max_val);
        output[i] = e;
        sum_exp += e;
    }
#endif
    for (; i < cols; i++) {
        float e = expf(input[i] - max_val);
        output[i] = e;
        sum_exp += e;
    }

    float inv_sum = 1.0f / sum_exp;
#if defined(OIL_AVX2) || defined(__AVX2__)
    __m256 inv = _mm256_set1_ps(inv_sum);
    for (i = 0; i + 8 <= cols; i += 8) {
        __m256 v = _mm256_loadu_ps(output + i);
        v = _mm256_mul_ps(v, inv);
        _mm256_storeu_ps(output + i, v);
    }
#endif
    for (; i < cols; i++) {
        output[i] *= inv_sum;
    }
}

// ========================================================================
// ScaledDotProductAttentionFunction: Q @ K^T / sqrt(D) -> softmax -> @ V
// Q: {B, H, S, D}, K/V: {B, KV_H, S_full, D}
// ========================================================================

class ScaledDotProductAttentionFunction : public AutogradFunction {
public:
    ScaledDotProductAttentionFunction(int64_t nh, int64_t nkv, int64_t hd)
        : num_heads_(nh), num_kv_heads_(nkv), head_dim_(hd),
          B_(0), H_(0), S_(0), D_(0), KV_H_(0), S_full_(0) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        const Tensor& Q = inputs[0];
        const Tensor& K = inputs[1];
        const Tensor& V = inputs[2];
        B_ = Q.dim(0); H_ = Q.dim(1); S_ = Q.dim(2); D_ = Q.dim(3);
        KV_H_ = K.dim(1); S_full_ = K.dim(2);

        float scale = 1.0f / std::sqrt((float)D_);

        Tensor score({B_, H_, S_, S_full_}, DType::F32);
        Tensor attn_weights({B_, H_, S_, S_full_}, DType::F32);
        Tensor attn_out({B_, H_, S_, D_}, DType::F32);

        const float* qd = Q.data<float>();
        const float* kd = K.data<float>();
        const float* vd = V.data<float>();
        float* sd = score.data<float>();
        float* wd = attn_weights.data<float>();
        float* od = attn_out.data<float>();

        // score = Q @ K^T * scale (+ causal mask for self-attention training)
        // When S_full == S we are in the training self-attention path: apply causal mask.
        const bool causal = (S_full_ == S_);
        for (int64_t b = 0; b < B_; b++) {
            for (int64_t h = 0; h < H_; h++) {
                int64_t kh = h % KV_H_;
                int64_t h_base = (b * H_ + h) * S_ * S_full_;
                int64_t kh_off = (b * KV_H_ + kh) * S_full_ * D_;
                for (int64_t s = 0; s < S_; s++) {
                    int64_t q_off = ((b * H_ + h) * S_ + s) * D_;
                    int64_t score_row = h_base + s * S_full_;
                    for (int64_t t = 0; t < S_full_; t++) {
                        if (causal && t > s) {
                            sd[score_row + t] = -FLT_MAX / 4.0f; // large negative, avoids NaN in exp
                            continue;
                        }
                        int64_t k_off = kh_off + t * D_;
                        float sum = 0;
                        for (int64_t d = 0; d < D_; d++)
                            sum += qd[q_off + d] * kd[k_off + d];
                        sd[score_row + t] = sum * scale;
                    }
                }
            }
        }

        // softmax (masked positions stay ~0 after exp of large negative)
        for (int64_t b = 0; b < B_; b++) {
            for (int64_t h = 0; h < H_; h++) {
                int64_t h_base = (b * H_ + h) * S_ * S_full_;
                for (int64_t s = 0; s < S_; s++) {
                    int64_t row = h_base + s * S_full_;
                    softmax_forward_avx2(wd + row, sd + row, S_full_);
                    if (causal) {
                        // Hard-zero future tokens for numerical cleanliness
                        for (int64_t t = s + 1; t < S_full_; t++)
                            wd[row + t] = 0.0f;
                        // Renormalize prefix so weights sum to 1
                        float z = 0.0f;
                        for (int64_t t = 0; t <= s && t < S_full_; t++) z += wd[row + t];
                        if (z > 0.0f) {
                            float inv = 1.0f / z;
                            for (int64_t t = 0; t <= s && t < S_full_; t++)
                                wd[row + t] *= inv;
                        }
                    }
                }
            }
        }

        // attn_out = attn_weights @ V
        for (int64_t b = 0; b < B_; b++) {
            for (int64_t h = 0; h < H_; h++) {
                int64_t kh = h % KV_H_;
                int64_t h_base = (b * H_ + h) * S_ * S_full_;
                int64_t kh_off = (b * KV_H_ + kh) * S_full_ * D_;
                for (int64_t s = 0; s < S_; s++) {
                    int64_t row = h_base + s * S_full_;
                    int64_t out_off = ((b * H_ + h) * S_ + s) * D_;
                    for (int64_t d = 0; d < D_; d++) {
                        float sum = 0;
                        for (int64_t t = 0; t < S_full_; t++)
                            sum += wd[row + t] * vd[kh_off + t * D_ + d];
                        od[out_off + d] = sum;
                    }
                }
            }
        }

        // Save Q, K, V, attn_weights for backward
        saved = inputs; // [0]=Q, [1]=K, [2]=V
        saved.push_back(attn_weights); // saved[3]

        return {attn_out};
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        const Tensor& d_out = grad_output[0]; // {B, H, S, D}
        const Tensor& Q = saved[0];
        const Tensor& K = saved[1];
        const Tensor& V = saved[2];
        const Tensor& attn_weights = saved[3]; // {B, H, S, S_full}

        const float* qd = Q.data<float>();
        const float* kd = K.data<float>();
        const float* vd = V.data<float>();
        const float* wd = attn_weights.data<float>();
        const float* dd = d_out.data<float>();

        Tensor dQ({B_, H_, S_, D_}, DType::F32);
        Tensor dK({B_, KV_H_, S_full_, D_}, DType::F32);
        Tensor dV({B_, KV_H_, S_full_, D_}, DType::F32);
        dQ.zero_(); dK.zero_(); dV.zero_();

        float scale = 1.0f / std::sqrt((float)D_);
        int64_t H = H_, KV_H = KV_H_, S = S_, S_full = S_full_, D = D_;
        int64_t B = B_;

        // Temporary: d_attn_weights
        Tensor d_aw({B, H, S, S_full}, DType::F32);
        float* daw = d_aw.data<float>();

        // d_attn_weights[b,h,s,t] = sum_d d_out[b,h,s,d] * V[b,kh,t,d]
        for (int64_t b = 0; b < B; b++) {
            for (int64_t h = 0; h < H; h++) {
                int64_t kh = h % KV_H;
                int64_t h_base = (b * H + h) * S * S_full;
                int64_t kh_off = (b * KV_H + kh) * S_full * D;
                for (int64_t s = 0; s < S; s++) {
                    int64_t row = h_base + s * S_full;
                    int64_t d_out_off = ((b * H + h) * S + s) * D;
                    for (int64_t t = 0; t < S_full; t++) {
                        float sum = 0;
                        for (int64_t d = 0; d < D; d++)
                            sum += dd[d_out_off + d] * vd[kh_off + t * D + d];
                        daw[row + t] = sum;
                    }
                }
            }
        }

        // d_score via softmax backward
        Tensor d_score({B, H, S, S_full}, DType::F32);
        float* ds = d_score.data<float>();
        for (int64_t b = 0; b < B; b++) {
            for (int64_t h = 0; h < H; h++) {
                int64_t h_base = (b * H + h) * S * S_full;
                for (int64_t s = 0; s < S; s++) {
                    int64_t row = h_base + s * S_full;
                    float dot = 0;
                    for (int64_t t = 0; t < S_full; t++)
                        dot += wd[row + t] * daw[row + t];
                    for (int64_t t = 0; t < S_full; t++)
                        ds[row + t] = wd[row + t] * (daw[row + t] - dot);
                }
            }
        }

        // dQ: {B, H, S, D}
        for (int64_t b = 0; b < B; b++) {
            for (int64_t h = 0; h < H; h++) {
                int64_t kh = h % KV_H;
                int64_t h_base = (b * H + h) * S * S_full;
                int64_t kh_off = (b * KV_H + kh) * S_full * D;
                for (int64_t s = 0; s < S; s++) {
                    int64_t row = h_base + s * S_full;
                    int64_t dq_off = ((b * H + h) * S + s) * D;
                    for (int64_t d = 0; d < D; d++) {
                        float sum = 0;
                        for (int64_t t = 0; t < S_full; t++)
                            sum += ds[row + t] * kd[kh_off + t * D + d];
                        dQ.data<float>()[dq_off + d] = sum * scale;
                    }
                }
            }
        }

        // dK: {B, KV_H, S_full, D}
        for (int64_t b = 0; b < B; b++) {
            for (int64_t h = 0; h < H; h++) {
                int64_t kh = h % KV_H;
                int64_t h_base = (b * H + h) * S * S_full;
                int64_t kh_off = (b * KV_H + kh) * S_full * D;
                for (int64_t t = 0; t < S_full; t++) {
                    for (int64_t d = 0; d < D; d++) {
                        float sum = 0;
                        for (int64_t s = 0; s < S; s++) {
                            int64_t row = h_base + s * S_full;
                            int64_t q_off = ((b * H + h) * S + s) * D;
                            sum += ds[row + t] * qd[q_off + d];
                        }
                        dK.data<float>()[kh_off + t * D + d] += sum * scale;
                    }
                }
            }
        }

        // dV: {B, KV_H, S_full, D}
        for (int64_t b = 0; b < B; b++) {
            for (int64_t h = 0; h < H; h++) {
                int64_t kh = h % KV_H;
                int64_t h_base = (b * H + h) * S * S_full;
                int64_t kh_off = (b * KV_H + kh) * S_full * D;
                for (int64_t t = 0; t < S_full; t++) {
                    for (int64_t d = 0; d < D; d++) {
                        float sum = 0;
                        for (int64_t s = 0; s < S; s++) {
                            int64_t row = h_base + s * S_full;
                            sum += wd[row + t] * dd[((b * H + h) * S + s) * D + d];
                        }
                        dV.data<float>()[kh_off + t * D + d] += sum;
                    }
                }
            }
        }

        return {dQ, dK, dV};
    }

private:
    int64_t num_heads_, num_kv_heads_, head_dim_;
    int64_t B_, H_, S_, D_, KV_H_, S_full_;
};

// ========================================================================
// BiasAddFunction: out = x + bias (broadcast over dim 0)
// ========================================================================

class BiasAddFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        const Tensor& x = inputs[0];
        const Tensor& bias = inputs[1];
        int64_t M = x.dim(0);
        int64_t N = x.dim(1);
        Tensor out(x.shape(), DType::F32);
        const float* xd = x.data<float>();
        const float* bd = bias.data<float>();
        float* od = out.data<float>();
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
                od[i * N + j] = xd[i * N + j] + bd[j];
        return {out};
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        const Tensor& d_out = grad_output[0];
        const Tensor& x = saved[0];
        int64_t N = x.dim(x.rank() - 1);
        int64_t numel = d_out.numel();
        const float* gd = d_out.data<float>();
        Tensor d_x(x.shape(), DType::F32);
        std::memcpy(d_x.data<float>(), gd, numel * sizeof(float));
        Tensor d_bias({N}, DType::F32);
        float* dbd = d_bias.data<float>();
        for (int64_t p = 0; p < numel; p++)
            dbd[p % N] += gd[p];
        return {d_x, d_bias};
    }
};

// ========================================================================
// EmbeddingFunction: embedding lookup (differentiable w.r.t. weight)
// ========================================================================

class EmbeddingFunction : public AutogradFunction {
public:
    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        const Tensor& ids = inputs[0];
        const Tensor& weight = inputs[1];
        int64_t N = ids.numel();
        int64_t D = weight.dim(1);
        int64_t V = weight.dim(0);
        Tensor out({N, D}, DType::F32);
        const float* id_d = ids.data<float>();
        const float* wd = weight.data<float>();
        float* od = out.data<float>();
        saved = {ids.clone(), weight}; // save for backward
        for (int64_t i = 0; i < N; i++) {
            int64_t token = (int64_t)id_d[i];
            if (token < 0 || token >= V) token = 0;
            std::memcpy(od + i * D, wd + token * D, D * sizeof(float));
        }
        return {out};
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        const Tensor& ids = saved[0];
        const Tensor& weight = saved[1];
        const Tensor& d_out = grad_output[0];
        int64_t N = ids.numel();
        int64_t D = weight.dim(1);
        int64_t V = weight.dim(0);
        const float* id_d = ids.data<float>();
        const float* gd = d_out.data<float>();
        Tensor d_weight(weight.shape(), DType::F32);
        d_weight.zero_();
        float* dwd = d_weight.data<float>();
        for (int64_t i = 0; i < N; i++) {
            int64_t token = (int64_t)id_d[i];
            if (token < 0 || token >= V) token = 0;
            for (int64_t d = 0; d < D; d++)
                dwd[token * D + d] += gd[i * D + d];
        }
        return {Tensor(), d_weight};
    }
};

// ========================================================================
// FlattenAttentionFunction: {B,H,S,D} → {B*S, H*D} with data reordering
// Needed because contiguous {B,H,S,D} cannot be direct-view-reshaped to {B*S, H*D}
// ========================================================================
class FlattenAttentionFunction : public AutogradFunction {
public:
    FlattenAttentionFunction(int64_t B, int64_t H, int64_t S, int64_t D)
        : B_(B), H_(H), S_(S), D_(D) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        const Tensor& x = inputs[0];
        Tensor out(Shape{B_ * S_, H_ * D_}, DType::F32);
        const float* xd = x.data<float>();
        float* od = out.data<float>();
        for (int64_t b = 0; b < B_; b++)
            for (int64_t h = 0; h < H_; h++)
                for (int64_t s = 0; s < S_; s++)
                    for (int64_t d = 0; d < D_; d++)
                        od[(b * S_ + s) * (H_ * D_) + h * D_ + d] =
                            xd[((b * H_ + h) * S_ + s) * D_ + d];
        return {out};
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        const Tensor& d_out = grad_output[0];
        Tensor d_x(Shape{B_, H_, S_, D_}, DType::F32);
        d_x.zero_();
        const float* dd = d_out.data<float>();
        float* dxd = d_x.data<float>();
        for (int64_t b = 0; b < B_; b++)
            for (int64_t h = 0; h < H_; h++)
                for (int64_t s = 0; s < S_; s++)
                    for (int64_t d = 0; d < D_; d++)
                        dxd[((b * H_ + h) * S_ + s) * D_ + d] =
                            dd[(b * S_ + s) * (H_ * D_) + h * D_ + d];
        return {d_x};
    }

private:
    int64_t B_, H_, S_, D_;
};

// ========================================================================
// TransposeFunction: permute two dims (differentiable)
// Forward: copies data to new layout; Backward: reverse permute
// ========================================================================

class TransposeFunction : public AutogradFunction {
public:
    TransposeFunction(int dim1, int dim2) : dim1_(dim1), dim2_(dim2) {}

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
        saved = inputs;
        return {inputs[0].transpose(dim1_, dim2_)};
    }

    std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) override {
        return {grad_output[0].transpose(dim1_, dim2_)};
    }

private:
    int dim1_, dim2_;
};

// ========================================================================
// AutogradEngine operation helpers
// ========================================================================

Tensor AutogradEngine::matmul_op(const Tensor& a, const Tensor& b, int64_t M, int64_t N, int64_t K) {
    if (!enabled_) {
        Tensor out({M, N}, DType::F32);
        kernel::scalar_gemm(
            a.data<float>(), b.data<float>(),
            out.data<float>(), (int)M, (int)N, (int)K
        );
        return out;
    }
    auto fn = std::make_shared<MatMulFunction>();
    auto outputs = fn->forward({a, b});
    Tensor& out = outputs[0];
    out.requires_grad(true);

    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {a, b};
    node->outputs = {out};
    instance().register_node(node);

    return out;
}

Tensor AutogradEngine::add_op(const Tensor& a, const Tensor& b) {
    if (!enabled_) {
        Tensor out(a.shape(), DType::F32);
        math::add(a, b, out);
        return out;
    }
    auto fn = std::make_shared<AddFunction>();
    auto outputs = fn->forward({a, b});
    Tensor& out = outputs[0];
    out.requires_grad(true);

    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {a, b};
    node->outputs = {out};
    instance().register_node(node);

    return out;
}

Tensor AutogradEngine::silu_op(const Tensor& x) {
    if (!enabled_) {
        Tensor out(x.shape(), DType::F32);
        math::silu(x, out);
        return out;
    }
    auto fn = std::make_shared<SiLUFunction>();
    auto outputs = fn->forward({x});
    Tensor& out = outputs[0];
    out.requires_grad(true);

    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x};
    node->outputs = {out};
    instance().register_node(node);

    return out;
}

Tensor AutogradEngine::mul_op(const Tensor& a, const Tensor& b) {
    if (!enabled_) {
        Tensor out(a.shape(), DType::F32);
        math::mul(a, b, out);
        return out;
    }
    auto fn = std::make_shared<MulFunction>();
    auto outputs = fn->forward({a, b});
    Tensor& out = outputs[0];
    out.requires_grad(true);

    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {a, b};
    node->outputs = {out};
    instance().register_node(node);

    return out;
}

Tensor AutogradEngine::cross_entropy_op(const Tensor& logits, const Tensor& labels) {
    auto fn = std::make_shared<CrossEntropyFunction>();
    auto outputs = fn->forward({logits, labels});
    Tensor& loss = outputs[0];

    if (enabled_) {
        auto node = std::make_shared<AutogradNode>();
        node->fn = fn;
        node->inputs = {logits, labels};
        node->outputs = {loss};
        loss.requires_grad(true);
        instance().register_node(node);
    }

    return loss;
}

Tensor AutogradEngine::rms_norm_op(const Tensor& x, const Tensor& gamma, float eps) {
    if (!enabled_) {
        Tensor out(x.shape(), DType::F32);
        math::rms_norm(x, gamma, eps, out);
        return out;
    }
    auto fn = std::make_shared<RMSNormFunction>();
    fn->saved_eps = eps;
    auto outputs = fn->forward({x, gamma});
    Tensor& out = outputs[0];
    out.requires_grad(true);

    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x, gamma};
    node->outputs = {out};
    instance().register_node(node);

    return out;
}

Tensor AutogradEngine::rotary_op(const Tensor& x, const Tensor& cos_cached,
                                  const Tensor& sin_cached,
                                  int64_t seq_start, int64_t seq_len) {
    if (!enabled_) {
        int64_t B = x.dim(0), H = x.dim(1), S = x.dim(2), D = x.dim(3);
        int64_t half = D / 2;
        Tensor out(x.shape(), DType::F32);
        const float* xd = x.data<float>();
        float* od = out.data<float>();
        const float* cos_d = cos_cached.data<float>();
        const float* sin_d = sin_cached.data<float>();
        for (int64_t b = 0; b < B; b++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t s = 0; s < S; s++) {
                    int64_t base = ((b * H + h) * S + s) * D;
                    int64_t pos = seq_start + s;
                    for (int64_t d = 0; d < half; d++) {
                        float x1 = xd[base + d], x2 = xd[base + d + half];
                        float cos_v = cos_d[pos * half + d];
                        float sin_v = sin_d[pos * half + d];
                        od[base + d] = x1 * cos_v - x2 * sin_v;
                        od[base + d + half] = x1 * sin_v + x2 * cos_v;
                    }
                }
        return out;
    }
    auto fn = std::make_shared<RotaryFunction>(x.dim(3), cos_cached, sin_cached, seq_start, seq_len);
    auto outputs = fn->forward({x});
    Tensor& out = outputs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x};
    node->outputs = {out};
    instance().register_node(node);
    return out;
}

Tensor AutogradEngine::attention_op(const Tensor& q, const Tensor& k, const Tensor& v,
                                     int64_t num_heads, int64_t num_kv_heads, int64_t head_dim) {
    if (!enabled_) {
        ScaledDotProductAttentionFunction fn(num_heads, num_kv_heads, head_dim);
        auto outputs = fn.forward({q, k, v});
        return outputs[0];
    }
    auto fn = std::make_shared<ScaledDotProductAttentionFunction>(num_heads, num_kv_heads, head_dim);
    auto outputs = fn->forward({q, k, v});
    Tensor& out = outputs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {q, k, v};
    node->outputs = {out};
    instance().register_node(node);
    return out;
}

Tensor AutogradEngine::bias_add_op(const Tensor& x, const Tensor& bias) {
    if (!enabled_) {
        int64_t M = x.dim(0), N = x.dim(1);
        Tensor out(x.shape(), DType::F32);
        const float* xd = x.data<float>();
        const float* bd = bias.data<float>();
        float* od = out.data<float>();
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++)
                od[i * N + j] = xd[i * N + j] + bd[j];
        return out;
    }
    auto fn = std::make_shared<BiasAddFunction>();
    auto outputs = fn->forward({x, bias});
    Tensor& out = outputs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x, bias};
    node->outputs = {out};
    instance().register_node(node);
    return out;
}

Tensor AutogradEngine::flatten_attention_op(const Tensor& x, int64_t B, int64_t H, int64_t S, int64_t D) {
    if (!enabled_) {
        Tensor out(Shape{B * S, H * D}, DType::F32);
        const float* xd = x.data<float>();
        float* od = out.data<float>();
        for (int64_t b = 0; b < B; b++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t s = 0; s < S; s++)
                    for (int64_t d = 0; d < D; d++)
                        od[(b * S + s) * (H * D) + h * D + d] =
                            xd[((b * H + h) * S + s) * D + d];
        return out;
    }
    auto fn = std::make_shared<FlattenAttentionFunction>(B, H, S, D);
    auto outputs = fn->forward({x});
    Tensor& out = outputs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x};
    node->outputs = {out};
    instance().register_node(node);
    return out;
}

Tensor AutogradEngine::transpose_op(const Tensor& x, int dim1, int dim2) {
    if (!enabled_) {
        return x.transpose(dim1, dim2);
    }
    auto fn = std::make_shared<TransposeFunction>(dim1, dim2);
    auto outputs = fn->forward({x});
    Tensor& out = outputs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {x};
    node->outputs = {out};
    instance().register_node(node);
    return out;
}

Tensor AutogradEngine::embedding_op(const Tensor& input_ids, const Tensor& weight) {
    if (!enabled_) {
        int64_t N = input_ids.numel();
        int64_t D = weight.dim(1);
        int64_t V = weight.dim(0);
        Tensor out({N, D}, DType::F32);
        const float* id_d = input_ids.data<float>();
        const float* wd = weight.data<float>();
        float* od = out.data<float>();
        for (int64_t i = 0; i < N; i++) {
            int64_t token = (int64_t)id_d[i];
            if (token < 0 || token >= V) token = 0;
            std::memcpy(od + i * D, wd + token * D, D * sizeof(float));
        }
        return out;
    }
    auto fn = std::make_shared<EmbeddingFunction>();
    auto outputs = fn->forward({input_ids, weight});
    Tensor& out = outputs[0];
    out.requires_grad(true);
    auto node = std::make_shared<AutogradNode>();
    node->fn = fn;
    node->inputs = {input_ids, weight};
    node->outputs = {out};
    instance().register_node(node);
    return out;
}

// ========================================================================
// Activation gradients
// ========================================================================

Tensor relu_grad(const Tensor& x, const Tensor& grad) {
    Tensor out(x.shape(), DType::F32);
    const float* xd = x.data<float>();
    const float* gd = grad.data<float>();
    float* od = out.data<float>();
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++)
        od[i] = (xd[i] > 0) ? gd[i] : 0.0f;
    return out;
}

Tensor silu_grad(const Tensor& x, const Tensor& grad) {
    Tensor out(x.shape(), DType::F32);
    const float* xd = x.data<float>();
    const float* gd = grad.data<float>();
    float* od = out.data<float>();
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) {
        float sig = 1.0f / (1.0f + std::exp(-xd[i]));
        od[i] = gd[i] * sig * (1.0f + xd[i] * (1.0f - sig));
    }
    return out;
}

Tensor gelu_grad(const Tensor& x, const Tensor& grad) {
    Tensor out(x.shape(), DType::F32);
    const float* xd = x.data<float>();
    const float* gd = grad.data<float>();
    float* od = out.data<float>();
    int64_t n = x.numel();
    float sqrt2_inv = 0.7071067811865475f;
    float sqrt_2pi_inv = 0.3989422804014327f;
    for (int64_t i = 0; i < n; i++) {
        float v = xd[i];
        float cdf = 0.5f * (1.0f + std::erf(v * sqrt2_inv));
        float pdf = std::exp(-0.5f * v * v) * sqrt_2pi_inv;
        od[i] = gd[i] * (cdf + v * pdf);
    }
    return out;
}

Tensor softmax_grad(const Tensor& output, const Tensor& grad) {
    Tensor out(output.shape(), DType::F32);
    const float* sd = output.data<float>();
    const float* gd = grad.data<float>();
    float* od = out.data<float>();
    int64_t rows = output.rank() >= 2 ? output.dim(0) : 1;
    int64_t cols = output.rank() >= 2 ? output.dim(output.rank() - 1) : output.numel();
    for (int64_t r = 0; r < rows; r++) {
        float dot = 0;
        for (int64_t c = 0; c < cols; c++)
            dot += sd[r * cols + c] * gd[r * cols + c];
        for (int64_t c = 0; c < cols; c++) {
            int64_t idx = r * cols + c;
            od[idx] = sd[idx] * (gd[idx] - dot);
        }
    }
    return out;
}

// ========================================================================
// Norm gradients
// ========================================================================

Tensor layer_norm_grad(const Tensor& x, const Tensor& gamma, const Tensor& grad, int N, Tensor* dgamma_out) {
    int64_t D = x.numel() / N;
    Tensor dx(x.shape());
    Tensor dgamma_local({D});
    dx.zero_(); dgamma_local.zero_();

    const float* xd = x.data<float>();
    const float* gd = grad.data<float>();
    const float* gam = gamma.data<float>();
    float* dxd = dx.data<float>();
    float* dgd = dgamma_out ? dgamma_out->data<float>() : dgamma_local.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        double mu = 0, var = 0;
        for (int64_t d = 0; d < D; ++d) mu += xd[n * D + d];
        mu /= D;
        for (int64_t d = 0; d < D; ++d) {
            double diff = xd[n * D + d] - mu;
            var += diff * diff;
        }
        var /= D;
        double inv_std = 1.0 / std::sqrt(var + 1e-5);

        double dnorm = 0, dvar = 0;
        for (int64_t d = 0; d < D; ++d) {
            double diff = xd[n * D + d] - mu;
            double dy = gd[n * D + d];
            dgd[d] += (float)(dy * inv_std);
            dnorm += dy * gam[d];
            dvar += dnorm * diff;
        }
        dvar *= -0.5 * inv_std * inv_std * inv_std / D;
        double dmu = 0;
        for (int64_t d = 0; d < D; ++d) {
            double diff = xd[n * D + d] - mu;
            double dy = gd[n * D + d];
            dmu -= dnorm * inv_std / D + 2 * dvar * diff / D;
        }
        for (int64_t d = 0; d < D; ++d) {
            double diff = xd[n * D + d] - mu;
            dxd[n * D + d] = (float)(dnorm * inv_std * gam[d] + 2 * dvar * diff + dmu / D);
        }
    }
    return dx;
}

Tensor rms_norm_grad(const Tensor& x, const Tensor& gamma, const Tensor& grad, int N, Tensor* dgamma) {
    int64_t D = x.numel() / N;
    Tensor dx(x.shape());
    dx.zero_();

    const float* xd = x.data<float>();
    const float* gd = grad.data<float>();
    const float* gam = gamma.data<float>();
    float* dxd = dx.data<float>();
    float* dgd = dgamma ? dgamma->data<float>() : nullptr;

    for (int64_t n = 0; n < N; ++n) {
        double ss = 0;
        for (int64_t d = 0; d < D; ++d) ss += (double)xd[n * D + d] * xd[n * D + d];
        ss /= D;
        double inv = 1.0 / std::sqrt(ss + 1e-5);
        double inv3 = inv * inv * inv;
        double sum_dy_gamma_x = 0;
        for (int64_t d = 0; d < D; ++d)
            sum_dy_gamma_x += (double)gd[n * D + d] * gam[d] * xd[n * D + d];
        sum_dy_gamma_x *= inv3 / D;
        for (int64_t d = 0; d < D; ++d) {
            dxd[n * D + d] = (float)((double)gd[n * D + d] * gam[d] * inv -
                                     sum_dy_gamma_x * (double)xd[n * D + d]);
        }
        if (dgd) {
            for (int64_t d = 0; d < D; ++d)
                dgd[d] += (float)((double)gd[n * D + d] * xd[n * D + d] * inv);
        }
    }
    return dx;
}

// ========================================================================
// Loss functions
// ========================================================================

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    OIL_CHECK(logits.numel() % targets.numel() == 0, "CE: shape mismatch");
    int64_t batch = targets.numel();
    int64_t C = logits.numel() / batch;
    const float* ld = logits.data<float>();
    const float* td = targets.data<float>();
    Tensor loss(Shape{1});
    float* ld_out = loss.data<float>();
    *ld_out = 0;
    for (int64_t b = 0; b < batch; b++) {
        float max_val = -INFINITY;
        for (int64_t c = 0; c < C; c++)
            if (ld[b * C + c] > max_val) max_val = ld[b * C + c];
        float sum_exp = 0;
        for (int64_t c = 0; c < C; c++)
            sum_exp += std::exp(ld[b * C + c] - max_val);
        float log_sum_exp = max_val + std::log(sum_exp);
        int64_t target = (int64_t)td[b];
        if (target < 0) target = 0;
        if (target >= C) target = C - 1;
        *ld_out += log_sum_exp - ld[b * C + target];
    }
    *ld_out /= (float)batch;
    return loss;
}

Tensor cross_entropy_grad(const Tensor& logits, const Tensor& targets) {
    int64_t batch = targets.numel();
    int64_t C = logits.numel() / batch;
    Tensor grad(logits.shape());
    const float* ld = logits.data<float>();
    const float* td = targets.data<float>();
    float* gd = grad.data<float>();
    for (int64_t b = 0; b < batch; b++) {
        float max_val = -INFINITY;
        for (int64_t c = 0; c < C; c++)
            if (ld[b * C + c] > max_val) max_val = ld[b * C + c];
        float sum_exp = 0;
        for (int64_t c = 0; c < C; c++)
            sum_exp += std::exp(ld[b * C + c] - max_val);
        int64_t target = (int64_t)td[b];
        if (target < 0) target = 0;
        if (target >= C) target = C - 1;
        for (int64_t c = 0; c < C; c++) {
            float soft = std::exp(ld[b * C + c] - max_val) / sum_exp;
            gd[b * C + c] = (soft - (c == target ? 1.0f : 0.0f)) / (float)batch;
        }
    }
    return grad;
}

// ========================================================================
// AutogradEngine
// ========================================================================

void AutogradEngine::register_parameter(Tensor* p) {
    if (p) param_map_[p->data()] = p;
}

void AutogradEngine::register_node(const std::shared_ptr<AutogradNode>& node) {
    if (next_is_checkpoint_) {
        node->checkpoint = true;
        node->fn->saved.clear();
        next_is_checkpoint_ = false;
        last_checkpoint_ = node;
    }
    nodes_.push_back(node);
    for (auto& out : node->outputs) {
        output_to_node_[out.data()] = node;
    }
}

void AutogradEngine::set_checkpoint() {
    instance().next_is_checkpoint_ = true;
}

bool AutogradEngine::is_checkpoint() {
    return instance().next_is_checkpoint_;
}

void AutogradEngine::backward(Tensor& loss) {
    if (!loss.has_grad() || loss.grad().numel() == 0) {
        Tensor g(loss.shape());
        g.fill(1.0f);
        loss.set_grad(g);
    }

    // Count consumers for each tensor (how many downstream inputs reference this data ptr)
    std::unordered_map<void*, int> consumer_count;
    for (auto& node : nodes_) {
        for (auto& inp : node->inputs) {
            if (inp.requires_grad())
                consumer_count[inp.data()]++;
        }
    }

    // The loss tensor's producer is consumed once (by the seed gradient)
    consumer_count[loss.data()]++;

    // Pending: how many more consumer gradients are needed before processing a producer
    std::unordered_map<void*, int> pending(consumer_count.begin(), consumer_count.end());

    // Accumulated gradients per data pointer (merge fan-in paths)
    std::unordered_map<void*, Tensor> accum;
    accum[loss.data()] = loss.grad();

    std::vector<const Tensor*> stack;
    std::unordered_set<const Tensor*> visited;
    stack.push_back(&loss);

    while (!stack.empty()) {
        const Tensor* t = stack.back();
        void* ptr = const_cast<void*>(t->data());
        auto pit = pending.find(ptr);
        if (pit == pending.end()) { stack.pop_back(); continue; }
        pit->second--;
        if (pit->second > 0) { stack.pop_back(); continue; }

        auto nit = output_to_node_.find(ptr);
        if (nit == output_to_node_.end() || nit->second.expired()) { stack.pop_back(); continue; }
        auto node = nit->second.lock();

        // Gradient checkpointing: re-run forward to regenerate intermediate values
        if (node->checkpoint && node->fn->saved.empty()) {
            node->fn->forward(node->inputs);
        }

        auto ait = accum.find(ptr);
        if (ait == accum.end()) { stack.pop_back(); continue; }
        auto grad_outputs = node->fn->backward({ ait->second });

        stack.pop_back();

        for (size_t i = 0; i < node->inputs.size() && i < grad_outputs.size(); ++i) {
            Tensor& inp = node->inputs[i];
            if (!inp.requires_grad()) continue;

            void* inp_ptr = const_cast<void*>(inp.data());

            auto pmit = param_map_.find(inp_ptr);
            if (pmit != param_map_.end()) {
                Tensor* orig = pmit->second;
                if (!orig->has_grad()) {
                    orig->set_grad(grad_outputs[i]);
                } else {
                    Tensor acc(orig->shape());
                    math::add(orig->grad(), grad_outputs[i], acc);
                    orig->set_grad(acc);
                }
            }

            auto aait = accum.find(inp_ptr);
            if (aait == accum.end()) {
                accum[inp_ptr] = grad_outputs[i];
            } else {
                Tensor tmp(grad_outputs[i].shape());
                math::add(aait->second, grad_outputs[i], tmp);
                aait->second = tmp;
            }

            if (visited.find(&inp) == visited.end()) {
                visited.insert(&inp);
                stack.push_back(&inp);
            }
        }
    }
}

void AutogradEngine::clear() {
    nodes_.clear();
    output_to_node_.clear();
    param_map_.clear();
}

AutogradEngine& AutogradEngine::instance() {
    static AutogradEngine engine;
    return engine;
}

} // namespace oil
