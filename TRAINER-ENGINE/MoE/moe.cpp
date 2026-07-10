#include "moe.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace oil {
namespace moe {

MoERouter::MoERouter(int64_t hidden_size, const MoEConfig& cfg)
    : config_(cfg)
{
    gate_weight = Tensor({hidden_size, cfg.num_experts});
    gate_weight.zero_();
}

RouterOutput MoERouter::forward(const Tensor& x) {
    int64_t B = x.dim(0);
    int64_t S = x.dim(1);
    int64_t hidden = x.dim(2);
    int64_t E = config_.num_experts;
    int64_t K = config_.top_k;
    int64_t T = B * S;

    Tensor logits({T, E});
    math::gemm(1.0f, x.reshape({T, hidden}), gate_weight, 0.0f, logits);

    Tensor probs({T, E});
    math::softmax(logits, probs, 1);

    Tensor indices({T, K});
    Tensor weights({T, K});
    float* p_ptr = probs.data<float>();
    float* w_ptr = weights.data<float>();
    int64_t* idx_ptr = indices.data<int64_t>();

    for (int64_t t = 0; t < T; ++t) {
        float* row = p_ptr + t * E;
        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(E);
        for (int64_t e = 0; e < E; ++e)
            scored.push_back({row[e], e});
        std::partial_sort(scored.begin(), scored.begin() + K, scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
        for (int64_t k = 0; k < K; ++k) {
            idx_ptr[t * K + k] = scored[k].second;
            w_ptr[t * K + k] = scored[k].first;
        }
    }

    float* g_ptr = probs.data<float>();
    std::vector<double> f_i(E, 0.0);
    std::vector<double> P_i(E, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = idx_ptr[t * K + k];
            f_i[e] += 1.0;
            P_i[e] += w_ptr[t * K + k];
        }
    }
    double load_balance = 0.0;
    for (int64_t e = 0; e < E; ++e) {
        f_i[e] /= (double)T;
        P_i[e] /= (double)T;
        load_balance += f_i[e] * P_i[e];
    }
    load_balance *= (double)E;

    RouterOutput out;
    out.gates = probs.reshape({B, S, E});
    out.indices = indices.reshape({B, S, K});
    out.weights = weights.reshape({B, S, K});
    out.load_balance_loss = (float)load_balance;
    return out;
}

MoEFFN::MoEFFN(const MoEConfig& cfg, const TransformerConfig& tcfg)
    : config_(cfg)
{
    experts.reserve(cfg.num_experts);
    for (int64_t i = 0; i < cfg.num_experts; ++i)
        experts.emplace_back(tcfg);
}

Tensor MoEFFN::forward(const Tensor& x, const RouterOutput& routing) {
    int64_t B = x.dim(0);
    int64_t S = x.dim(1);
    int64_t hidden = x.dim(2);
    int64_t T = B * S;
    int64_t K = config_.top_k;
    int64_t E = config_.num_experts;

    Tensor output({T, hidden});
    output.zero_();

    const int64_t* idx_ptr = routing.indices.data<int64_t>();
    const float* w_ptr = routing.weights.data<float>();

    Tensor x_flat = x.reshape({T, hidden});
    const float* x_ptr = x_flat.data<float>();
    float* o_ptr = output.data<float>();

    float z_loss = 0.0f;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = idx_ptr[t * K + k];
            float weight = w_ptr[t * K + k];

            Tensor token_in({1, hidden});
            std::memcpy(token_in.data<float>(), x_ptr + t * hidden, hidden * sizeof(float));

            Tensor expert_out = experts[e].forward(token_in);

            for (int64_t h = 0; h < hidden; ++h)
                o_ptr[t * hidden + h] += weight * expert_out.data<float>()[h];

            for (int64_t h = 0; h < hidden; ++h)
                z_loss += expert_out.data<float>()[h] * expert_out.data<float>()[h];
        }
    }
    z_loss = config_.z_loss_coef * z_loss / (float)T;

    Tensor out = output.reshape({B, S, hidden});
    return out;
}

MoEBlock::MoEBlock(const MoEConfig& moe_cfg, const TransformerConfig& tcfg)
    : attention_norm(tcfg.hidden_size, tcfg.norm_eps),
      attention(tcfg),
      moe_norm(tcfg.hidden_size, tcfg.norm_eps),
      router(tcfg.hidden_size, moe_cfg),
      moe_ffn(moe_cfg, tcfg)
{
}

Tensor MoEBlock::forward(const Tensor& x, const Tensor& positions,
                          const Tensor& mask, KVCache& cache, int layer_idx) {
    Tensor attn_normed = attention_norm.forward(x);
    Tensor attn_out = attention.forward(attn_normed, positions, mask, cache, layer_idx);
    Tensor residual({x.shape()});
    math::add(x, attn_out, residual);

    Tensor moe_normed = moe_norm.forward(residual);
    RouterOutput rout = router.forward(moe_normed);
    Tensor moe_out = moe_ffn.forward(moe_normed, rout);

    Tensor result({residual.shape()});
    math::add(residual, moe_out, result);
    return result;
}

} // namespace moe
} // namespace oil
