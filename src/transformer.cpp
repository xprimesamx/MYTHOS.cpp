#include "oil/transformer.h"
#include "oil/math.h"
#include "oil/autograd.h"
#include "oil/random.h"
#include <cmath>
#include <cstring>

#if defined(OIL_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace oil {

// Initialize weight with uniform random values in [-bound, bound]
static void init_uniform(Tensor& t, float bound) {
    RNG rng(42);
    float* d = t.data<float>();
    for (int64_t i = 0; i < t.numel(); i++)
        d[i] = (rng.uniform() * 2.0f - 1.0f) * bound;
}

// Embedding
Embedding::Embedding(int64_t vocab_size, int64_t dim)
    : weight(Tensor::zeros(Shape{vocab_size, dim})) {
    float scale = 1.0f / std::sqrt((float)dim);
    init_uniform(weight, scale);
}

Tensor Embedding::forward(const Tensor& input_ids) const {
    if (AutogradEngine::enabled())
        return AutogradEngine::embedding_op(input_ids, weight);
    int64_t batch = input_ids.numel();
    int64_t dim = weight.shape().dims[1];
    int64_t vocab = weight.shape().dims[0];
    Tensor out(Shape{batch, dim}, DType::F32);
    const float* ids = input_ids.data<float>();
    const float* w = weight.data<float>();
    float* od = out.data<float>();
    for (int64_t i = 0; i < batch; i++) {
        int64_t id = (int64_t)ids[i];
        if (id < 0) id = 0;
        if (id >= vocab) id = 0;
        memcpy(od + i * dim, w + id * dim, dim * sizeof(float));
    }
    return out;
}

size_t Embedding::param_count() const { return weight.numel(); }

// Linear
Linear::Linear(int64_t in_features, int64_t out_features)
    : weight(Tensor::zeros(Shape{out_features, in_features})),
      bias(Tensor::zeros(Shape{out_features})) {
    float scale = 1.0f / std::sqrt((float)in_features);
    init_uniform(weight, scale);
}

Tensor Linear::forward(const Tensor& input) const {
    int64_t in_dim = weight.shape().dims[1];
    int64_t out_dim = weight.shape().dims[0];
    int64_t in_rank = input.rank();
    int64_t batch = input.numel() / in_dim;
    
    Tensor inp2d = (in_rank > 2) ? input.reshape(Shape{batch, in_dim}) : input;
    
    Tensor out = AutogradEngine::matmul_op(inp2d, weight, batch, out_dim, in_dim);
    
    if (bias.numel() > 0) {
        if (AutogradEngine::enabled()) {
            out = AutogradEngine::bias_add_op(out, bias);
        } else {
            float* od = (float*)out.data();
            const float* bd = (const float*)bias.data();
            for (int64_t i = 0; i < batch; i++) {
                for (int64_t j = 0; j < out_dim; j++) {
                    od[i * out_dim + j] += bd[j];
                }
            }
        }
    }
    
    // Restore leading dims if input was > 2D
    if (in_rank > 2) {
        Shape out_shape = input.shape();
        out_shape.dims[in_rank - 1] = out_dim;
        return out.reshape(out_shape);
    }
    return out;
}

size_t Linear::param_count() const { return weight.numel() + bias.numel(); }

// RMSNorm
RMSNorm::RMSNorm(int64_t size, float eps_val)
    : weight(Tensor::ones(Shape{size})), eps(eps_val) {}

Tensor RMSNorm::forward(const Tensor& input) const {
    return AutogradEngine::rms_norm_op(input, weight, eps);
}

// RotaryEmbedding
RotaryEmbedding::RotaryEmbedding(int64_t hd, int64_t max_seq_len, float t)
    : head_dim(hd), theta(t) {
    cos_cached = Tensor(Shape{max_seq_len, hd / 2}, DType::F32);
    sin_cached = Tensor(Shape{max_seq_len, hd / 2}, DType::F32);
    float* cos_d = (float*)cos_cached.data();
    float* sin_d = (float*)sin_cached.data();
    for (int64_t i = 0; i < max_seq_len; i++) {
        for (int64_t j = 0; j < hd / 2; j++) {
            float inv_freq = 1.0f / std::pow(theta, (float)(2 * j) / hd);
            float val = (float)i * inv_freq;
            cos_d[i * hd / 2 + j] = std::cos(val);
            sin_d[i * hd / 2 + j] = std::sin(val);
        }
    }
}

void RotaryEmbedding::apply(Tensor& x, int64_t seq_start, int64_t seq_len) const {
    int64_t B = x.shape().dims[0];
    int64_t H = x.shape().dims[1];
    int64_t S = x.shape().dims[2];
    int64_t D = x.shape().dims[3];
    float* xd = (float*)x.data();
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t s = 0; s < S; s++) {
                int64_t pos = seq_start + s;
                for (int64_t d = 0; d < D / 2; d++) {
                    float x1 = xd[b * H * S * D + h * S * D + s * D + d];
                    float x2 = xd[b * H * S * D + h * S * D + s * D + d + D / 2];
                    float cos_v = ((float*)cos_cached.data())[pos * D / 2 + d];
                    float sin_v = ((float*)sin_cached.data())[pos * D / 2 + d];
                    xd[b * H * S * D + h * S * D + s * D + d] = x1 * cos_v - x2 * sin_v;
                    xd[b * H * S * D + h * S * D + s * D + d + D / 2] = x1 * sin_v + x2 * cos_v;
                }
            }
        }
    }
}

// Attention
Attention::Attention(const TransformerConfig& cfg)
    : num_heads(cfg.num_heads), 
      num_kv_heads(cfg.num_kv_heads > 0 ? cfg.num_kv_heads : cfg.num_heads),
      head_dim(cfg.head_dim),
      rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
{
    q_proj = Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim);
    k_proj = Linear(cfg.hidden_size, num_kv_heads * cfg.head_dim);
    v_proj = Linear(cfg.hidden_size, num_kv_heads * cfg.head_dim);
    o_proj = Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size);
}

Tensor Attention::forward(const Tensor& x, const Tensor& positions,
                           const Tensor& mask, KVCache& cache, int layer_idx) const {
    int64_t B = x.shape().dims[0];
    int64_t S = x.shape().dims[1];
    
    Tensor q = q_proj.forward(x);
    Tensor k = k_proj.forward(x);
    Tensor v = v_proj.forward(x);
    
    // Reshape to [B, H, S, head_dim]
    Tensor q_reshaped = q.reshape(Shape{B, S, num_heads, head_dim});
    Tensor k_reshaped = k.reshape(Shape{B, S, num_kv_heads, head_dim});
    Tensor v_reshaped = v.reshape(Shape{B, S, num_kv_heads, head_dim});
    
    Tensor attn_out;
    if (AutogradEngine::enabled()) {
        // Training path: use autograd RoPE and attention (no KV cache)
        Tensor q_t = AutogradEngine::transpose_op(q_reshaped, 1, 2);
        Tensor k_t = AutogradEngine::transpose_op(k_reshaped, 1, 2);
        Tensor v_t = AutogradEngine::transpose_op(v_reshaped, 1, 2);
        Tensor q_rope = AutogradEngine::rotary_op(q_t, rope.cos_cached, rope.sin_cached, 0, S);
        Tensor k_rope = AutogradEngine::rotary_op(k_t, rope.cos_cached, rope.sin_cached, 0, S);
        attn_out = AutogradEngine::attention_op(q_rope, k_rope, v_t, num_heads, num_kv_heads, head_dim);
    } else {
        // Inference path: transpose {B,S,H,D} -> {B,H,S,D}, then RoPE + KV cache + attention
        Tensor q_t = q_reshaped.transpose(1, 2);  // {B, H, S, D}
        Tensor k_t = k_reshaped.transpose(1, 2);  // {B, KV_H, S, D}
        Tensor v_t = v_reshaped.transpose(1, 2);  // {B, KV_H, S, D}
        
        int64_t seq_start = cache.context_len();
        rope.apply(q_t, seq_start, S);
        rope.apply(k_t, seq_start, S);
        cache.append(layer_idx, k_t, v_t);
        auto [k_full, v_full] = cache.get_all(layer_idx);
        int64_t S_full = k_full.shape().dims[2];
        float scale = 1.0f / std::sqrt((float)head_dim);
        
        const float* qd = (const float*)q_t.data();
        const float* kd = (const float*)k_full.data();
        const float* vd = (const float*)v_full.data();
        
        Tensor score(Shape{B, num_heads, S, S_full}, DType::F32);
        float* sd = (float*)score.data();
        for (int64_t b = 0; b < B; b++) {
            for (int64_t h = 0; h < num_heads; h++) {
                int64_t kh = h % num_kv_heads;
                int64_t q_base = ((b * num_heads + h) * S) * head_dim;
                int64_t k_base = kh * S_full * head_dim;
                int64_t s_base = (b * num_heads + h) * S * S_full;
                for (int64_t s = 0; s < S; s++) {
                    const float* qptr = qd + q_base + s * head_dim;
                    for (int64_t t = 0; t < S_full; t++) {
                        const float* kptr = kd + k_base + t * head_dim;
                        float sum = 0;
#if defined(OIL_AVX2) || defined(__AVX2__)
                        __m256 sumv = _mm256_setzero_ps();
                        int64_t d;
                        for (d = 0; d + 8 <= head_dim; d += 8) {
                            __m256 qv = _mm256_loadu_ps(qptr + d);
                            __m256 kv = _mm256_loadu_ps(kptr + d);
                            sumv = _mm256_fmadd_ps(qv, kv, sumv);
                        }
                        float hsum[8];
                        _mm256_storeu_ps(hsum, sumv);
                        sum = hsum[0] + hsum[1] + hsum[2] + hsum[3]
                            + hsum[4] + hsum[5] + hsum[6] + hsum[7];
                        for (; d < head_dim; d++)
                            sum += qptr[d] * kptr[d];
#else
                        for (int64_t d = 0; d < head_dim; d++)
                            sum += qptr[d] * kptr[d];
#endif
                        sd[s_base + s * S_full + t] = sum * scale;
                    }
                }
            }
        }
        
        Tensor attn_weights(score.shape(), DType::F32);
        float* wd = (float*)attn_weights.data();
        for (int64_t b = 0; b < B; b++) {
            for (int64_t h = 0; h < num_heads; h++) {
                int64_t base = (b * num_heads + h) * S * S_full;
                for (int64_t s = 0; s < S; s++) {
                    int64_t row = base + s * S_full;
#if defined(OIL_AVX2) || defined(__AVX2__)
                    __m256 maxv = _mm256_loadu_ps(sd + row);
                    int64_t t = 8;
                    for (; t + 8 <= S_full; t += 8) {
                        __m256 v = _mm256_loadu_ps(sd + row + t);
                        maxv = _mm256_max_ps(maxv, v);
                    }
                    float max_arr[8];
                    _mm256_storeu_ps(max_arr, maxv);
                    float max_v = max_arr[0];
                    for (int j = 1; j < 8; j++) if (max_arr[j] > max_v) max_v = max_arr[j];
                    for (t = (t >= S_full ? S_full - 8 : t); t < S_full; t++)
                        if (sd[row + t] > max_v) max_v = sd[row + t];

                    __m256 sumv = _mm256_setzero_ps();
                    __m256 mvec = _mm256_set1_ps(max_v);
                    for (t = 0; t + 8 <= S_full; t += 8) {
                        __m256 sv = _mm256_loadu_ps(sd + row + t);
                        sv = _mm256_sub_ps(sv, mvec);
                        __m256 x = sv;
                        __m256 e = _mm256_set1_ps(1.0f);
                        __m256 term = x;
                        e = _mm256_add_ps(e, term);
                        term = _mm256_mul_ps(term, x);
                        e = _mm256_add_ps(e, _mm256_mul_ps(term, _mm256_set1_ps(1.0f/2.0f)));
                        term = _mm256_mul_ps(term, x);
                        e = _mm256_add_ps(e, _mm256_mul_ps(term, _mm256_set1_ps(1.0f/6.0f)));
                        term = _mm256_mul_ps(term, x);
                        e = _mm256_add_ps(e, _mm256_mul_ps(term, _mm256_set1_ps(1.0f/24.0f)));
                        _mm256_storeu_ps(wd + row + t, e);
                        sumv = _mm256_add_ps(sumv, e);
                    }
                    float hsum[8];
                    _mm256_storeu_ps(hsum, sumv);
                    float sum_exp = hsum[0] + hsum[1] + hsum[2] + hsum[3]
                                  + hsum[4] + hsum[5] + hsum[6] + hsum[7];
                    for (; t < S_full; t++) {
                        float e = std::exp(sd[row + t] - max_v);
                        wd[row + t] = e;
                        sum_exp += e;
                    }
                    float inv_sum = 1.0f / sum_exp;
                    __m256 iv = _mm256_set1_ps(inv_sum);
                    for (t = 0; t + 8 <= S_full; t += 8) {
                        __m256 wv = _mm256_loadu_ps(wd + row + t);
                        wv = _mm256_mul_ps(wv, iv);
                        _mm256_storeu_ps(wd + row + t, wv);
                    }
                    for (; t < S_full; t++)
                        wd[row + t] *= inv_sum;
#else
                    float max_v = -INFINITY;
                    for (int64_t t = 0; t < S_full; t++)
                        if (sd[row + t] > max_v) max_v = sd[row + t];
                    float sum_exp = 0;
                    for (int64_t t = 0; t < S_full; t++) {
                        float e = std::exp(sd[row + t] - max_v);
                        wd[row + t] = e;
                        sum_exp += e;
                    }
                    float inv_sum = 1.0f / sum_exp;
                    for (int64_t t = 0; t < S_full; t++)
                        wd[row + t] *= inv_sum;
#endif
                }
            }
        }
        
        attn_out = Tensor(Shape{B, num_heads, S, head_dim}, DType::F32);
        float* aod = (float*)attn_out.data();
        for (int64_t b = 0; b < B; b++) {
            for (int64_t h = 0; h < num_heads; h++) {
                int64_t kh = h % num_kv_heads;
                int64_t w_base = (b * num_heads + h) * S * S_full;
                int64_t v_base = kh * S_full * head_dim;
                int64_t o_base = ((b * num_heads + h) * S) * head_dim;
                for (int64_t s = 0; s < S; s++) {
                    for (int64_t d = 0; d < head_dim; d++) {
                        float sum = 0;
                        const float* wptr = wd + w_base + s * S_full;
                        const float* vptr = vd + v_base + d;
#if defined(OIL_AVX2) || defined(__AVX2__)
                        __m256 sumv = _mm256_setzero_ps();
                        int64_t t;
                        for (t = 0; t + 8 <= S_full; t += 8) {
                            __m256 wv = _mm256_loadu_ps(wptr + t);
                            __m256 vv = _mm256_loadu_ps(vptr + t * head_dim);
                            sumv = _mm256_fmadd_ps(wv, vv, sumv);
                        }
                        float hsum[8];
                        _mm256_storeu_ps(hsum, sumv);
                        sum = hsum[0] + hsum[1] + hsum[2] + hsum[3]
                            + hsum[4] + hsum[5] + hsum[6] + hsum[7];
                        for (; t < S_full; t++)
                            sum += wptr[t] * vptr[t * head_dim + d];
#else
                        for (int64_t t = 0; t < S_full; t++)
                            sum += wptr[t] * vptr[t * head_dim + d];
#endif
                        aod[o_base + s * head_dim + d] = sum;
                    }
                }
            }
        }
    }
    
    // Output projection: flatten {B,H,S,D} -> {B*S, H*D}
    Tensor attn_flat;
    if (AutogradEngine::enabled()) {
        attn_flat = AutogradEngine::flatten_attention_op(attn_out, B, num_heads, S, head_dim);
    } else {
        // Transpose {B,H,S,D} -> {B,S,H,D} then flatten
        Tensor attn_t = attn_out.transpose(1, 2);
        attn_flat = attn_t.reshape(Shape{B * S, num_heads * head_dim});
    }
    Tensor o_out = o_proj.forward(attn_flat);
    return o_out.reshape(Shape{B, S, o_out.dim(1)});
}

// FFN
FFN::FFN(const TransformerConfig& cfg)
    : activation(cfg.activation)
{
    gate_proj = Linear(cfg.hidden_size, cfg.ffn_hidden_size);
    up_proj = Linear(cfg.hidden_size, cfg.ffn_hidden_size);
    down_proj = Linear(cfg.ffn_hidden_size, cfg.hidden_size);
}

Tensor FFN::forward(const Tensor& x) const {
    Tensor gate = gate_proj.forward(x);
    Tensor up = up_proj.forward(x);
    
    if (activation == Activation::SiLU) {
        gate = AutogradEngine::silu_op(gate);
    } else if (activation == Activation::GELU) {
        math::gelu(gate, gate);
    } else {
        math::relu(gate, gate);
    }
    
    Tensor hidden = AutogradEngine::mul_op(gate, up);
    return down_proj.forward(hidden);
}

// TransformerBlock
TransformerBlock::TransformerBlock(const TransformerConfig& cfg)
    : attention_norm(cfg.hidden_size, cfg.norm_eps),
      attention(cfg),
      ffn_norm(cfg.hidden_size, cfg.norm_eps),
      ffn(cfg) {}

Tensor TransformerBlock::forward(const Tensor& x, const Tensor& positions,
                                  const Tensor& mask, KVCache& cache, int layer_idx) const {
    Tensor attn_input = attention_norm.forward(x);
    Tensor attn_out = attention.forward(attn_input, positions, mask, cache, layer_idx);
    attn_out = AutogradEngine::add_op(attn_out, x);
    
    Tensor ffn_input = ffn_norm.forward(attn_out);
    Tensor ffn_out = ffn.forward(ffn_input);
    ffn_out = AutogradEngine::add_op(ffn_out, attn_out);
    return ffn_out;
}

} // namespace oil
