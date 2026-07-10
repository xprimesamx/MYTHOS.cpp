#include "oil/transformer.h"
#include "oil/math.h"
#include <cmath>
#include <cstring>

namespace oil {

// Embedding
Embedding::Embedding(int64_t vocab_size, int64_t dim)
    : weight(Tensor::zeros(Shape{vocab_size, dim})) {}

Tensor Embedding::forward(const Tensor& input_ids) const {
    int64_t batch = input_ids.numel();
    int64_t dim = weight.shape().dims[1];
    Tensor out(Shape{batch, dim}, DType::F32);
    const int* ids = (const int*)input_ids.data();
    const float* w = (const float*)weight.data();
    float* od = (float*)out.data();
    for (int64_t i = 0; i < batch; i++) {
        int id = ids[i];
        if (id < 0) id = 0;
        if (id >= weight.shape().dims[0]) id = 0;
        memcpy(od + i * dim, w + id * dim, dim * sizeof(float));
    }
    return out;
}

size_t Embedding::param_count() const { return weight.numel(); }

// Linear
Linear::Linear(int64_t in_features, int64_t out_features)
    : weight(Tensor::zeros(Shape{out_features, in_features})),
      bias(Tensor::zeros(Shape{out_features})) {}

Tensor Linear::forward(const Tensor& input) const {
    int64_t in_dim = weight.shape().dims[1];
    int64_t out_dim = weight.shape().dims[0];
    int64_t in_rank = input.rank();
    int64_t batch = input.numel() / in_dim;
    
    // Flatten leading dims for the gemm
    Tensor inp2d = (in_rank > 2) ? input.reshape(Shape{batch, in_dim}) : input;
    
    Tensor out(Shape{batch, out_dim}, DType::F32);
    kernel::scalar_gemm(
        (const float*)inp2d.data(),
        (const float*)weight.data(),
        (float*)out.data(),
        batch, out_dim, in_dim
    );
    
    if (bias.numel() > 0) {
        float* od = (float*)out.data();
        const float* bd = (const float*)bias.data();
        for (int64_t i = 0; i < batch; i++) {
            for (int64_t j = 0; j < out_dim; j++) {
                od[i * out_dim + j] += bd[j];
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
    Tensor out = input.clone();
    math::rms_norm(input, weight, eps, out);
    return out;
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
    int64_t D = x.shape().dims[2];
    
    Tensor q = q_proj.forward(x);
    Tensor k = k_proj.forward(x);
    Tensor v = v_proj.forward(x);
    
    // Reshape to [B, H, S, head_dim]
    Tensor q_reshaped = q.reshape(Shape{B, S, num_heads, head_dim}).transpose(1, 2);
    Tensor k_reshaped = k.reshape(Shape{B, S, num_kv_heads, head_dim}).transpose(1, 2);
    Tensor v_reshaped = v.reshape(Shape{B, S, num_kv_heads, head_dim}).transpose(1, 2);
    
    // Apply RoPE
    int64_t seq_start = cache.context_len();
    rope.apply(q_reshaped, seq_start, S);
    rope.apply(k_reshaped, seq_start, S);
    
    // Update KV cache
    cache.append(layer_idx, k_reshaped, v_reshaped);
    
    // Get full K,V from cache
    auto [k_full, v_full] = cache.get_all(layer_idx);
    
    // Scaled dot-product attention
    int64_t S_full = k_full.shape().dims[2];
    float scale = 1.0f / std::sqrt((float)head_dim);
    Tensor score(Shape{B, num_heads, S, S_full}, DType::F32);
    
    const float* qd = (const float*)q_reshaped.data();
    const float* kd = (const float*)k_full.data();
    float* sd = (float*)score.data();
    
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < num_heads; h++) {
            int64_t kh = h % num_kv_heads;
            for (int64_t s = 0; s < S; s++) {
                for (int64_t t = 0; t < S_full; t++) {
                    float sum = 0;
                    for (int64_t d = 0; d < head_dim; d++) {
                        sum += qd[b * num_heads * S * head_dim + h * S * head_dim + s * head_dim + d]
                             * kd[kh * S_full * head_dim + t * head_dim + d];  // kv cache is shared (batch=1)
                    }
                    sd[b * num_heads * S * S_full + h * S * S_full + s * S_full + t] = sum * scale;
                }
            }
        }
    }
    
    // Softmax
    Tensor attn_weights(score.shape(), DType::F32);
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < num_heads; h++) {
            for (int64_t s = 0; s < S; s++) {
                float max_v = -INFINITY;
                for (int64_t t = 0; t < S_full; t++) {
                    float v = sd[b * num_heads * S * S_full + h * S * S_full + s * S_full + t];
                    if (v > max_v) max_v = v;
                }
                float sum_exp = 0;
                for (int64_t t = 0; t < S_full; t++) {
                    float e = std::exp(sd[b * num_heads * S * S_full + h * S * S_full + s * S_full + t] - max_v);
                    sum_exp += e;
                    attn_weights.data<float>()[b * num_heads * S * S_full + h * S * S_full + s * S_full + t] = e;
                }
                for (int64_t t = 0; t < S_full; t++) {
                    attn_weights.data<float>()[b * num_heads * S * S_full + h * S * S_full + s * S_full + t] /= sum_exp;
                }
            }
        }
    }
    
    // Attention output
    Tensor attn_out(Shape{B, num_heads, S, head_dim}, DType::F32);
    const float* aw = (const float*)attn_weights.data();
    const float* vd = (const float*)v_full.data();
    float* aod = (float*)attn_out.data();
    
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < num_heads; h++) {
            int64_t kh = h % num_kv_heads;
            for (int64_t s = 0; s < S; s++) {
                for (int64_t d = 0; d < head_dim; d++) {
                    float sum = 0;
                    for (int64_t t = 0; t < S_full; t++) {
                        sum += aw[b * num_heads * S * S_full + h * S * S_full + s * S_full + t]
                             * vd[kh * S_full * head_dim + t * head_dim + d];  // kv cache is shared (batch=1)
                    }
                    aod[b * num_heads * S * head_dim + h * S * head_dim + s * head_dim + d] = sum;
                }
            }
        }
    }
    
    // Output projection: flatten B,S for gemm then restore shape
    Tensor attn_t = attn_out.transpose(1, 2);
    Tensor attn_flat = attn_t.reshape(Shape{B * S, num_heads * head_dim});
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
        math::silu(gate, gate);
    } else if (activation == Activation::GELU) {
        math::gelu(gate, gate);
    } else {
        math::relu(gate, gate);
    }
    
    Tensor hidden(Shape{x.shape().dims[0], x.shape().dims[1], gate_proj.weight.shape().dims[0]}, DType::F32);
    math::mul(gate, up, hidden);
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
    Tensor residual = x.clone();
    Tensor attn_input = attention_norm.forward(x.clone());
    Tensor attn_out = attention.forward(attn_input, positions, mask, cache, layer_idx);
    math::add(attn_out, residual, attn_out);
    
    Tensor ffn_input = ffn_norm.forward(attn_out.clone());
    Tensor ffn_out = ffn.forward(ffn_input);
    math::add(ffn_out, attn_out, ffn_out);
    return ffn_out;
}

} // namespace oil
