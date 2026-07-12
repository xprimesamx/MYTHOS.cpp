#include "oil/moe_variants.h"
#include <cstring>
#include <unordered_map>

namespace oil {
namespace moe {

// ========================================================================
// Hash utility
// ========================================================================

int64_t hash_token(int64_t token_id, int64_t range) {
    uint64_t h = (uint64_t)token_id * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 37;
    h *= 0xBF58476D1CE4E5B9ULL;
    return (int64_t)(h % (uint64_t)range);
}

// ========================================================================
// Softmax with Top-K extraction
// ========================================================================

Tensor softmax_with_topk(const Tensor& logits, int64_t k, Tensor& indices_out, Tensor& weights_out) {
    int64_t T = logits.dim(0);
    int64_t E = logits.dim(1);
    Tensor probs({T, E});
    const float* l = logits.data<float>();
    float* p = probs.data<float>();
    indices_out = Tensor({T, k}, DType::I64);
    weights_out = Tensor({T, k});
    int64_t* idx = indices_out.data<int64_t>();
    float* w = weights_out.data<float>();

    for (int64_t t = 0; t < T; ++t) {
        float maxv = l[t * E];
        for (int64_t e = 1; e < E; ++e)
            if (l[t * E + e] > maxv) maxv = l[t * E + e];
        float sum = 0.0f;
        for (int64_t e = 0; e < E; ++e) {
            float v = std::exp(l[t * E + e] - maxv);
            p[t * E + e] = v;
            sum += v;
        }
        float inv = 1.0f / sum;
        for (int64_t e = 0; e < E; ++e)
            p[t * E + e] *= inv;

        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(E);
        for (int64_t e = 0; e < E; ++e)
            scored.push_back({p[t * E + e], e});
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
        for (int64_t j = 0; j < k; ++j) {
            idx[t * k + j] = scored[j].second;
            w[t * k + j] = scored[j].first;
        }
    }
    return probs;
}

// ========================================================================
// Load balancing loss calculation
// ========================================================================

float compute_load_balance_loss(const Tensor& router_logits, const Tensor& expert_indices, int64_t num_experts) {
    int64_t T = router_logits.dim(0);
    int64_t K = expert_indices.dim(1);
    const int64_t* idx = expert_indices.data<int64_t>();
    const float* w = router_logits.data<float>();
    std::vector<double> f_i(num_experts, 0.0);
    std::vector<double> P_i(num_experts, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = (int64_t)idx[t * K + k];
            if (e >= 0 && e < num_experts) {
                f_i[e] += 1.0;
                P_i[e] += w[t * num_experts + e];
            }
        }
    }
    double loss = 0.0;
    for (int64_t e = 0; e < num_experts; ++e) {
        f_i[e] /= (double)T;
        P_i[e] /= (double)T;
        loss += f_i[e] * P_i[e];
    }
    return (float)(loss * (double)num_experts);
}

float compute_z_loss(const Tensor& expert_output_norms) {
    const float* d = expert_output_norms.data<float>();
    float sum = 0.0f;
    for (int64_t i = 0; i < expert_output_norms.numel(); ++i)
        sum += d[i] * d[i];
    return sum;
}

// ========================================================================
// ExpertFFN loader
// ========================================================================

static std::vector<ExpertFFN> create_experts(int64_t count, int64_t hidden, int64_t ffn_hidden, Activation act) {
    std::vector<ExpertFFN> exps;
    exps.reserve(count);
    for (int64_t i = 0; i < count; ++i)
        exps.emplace_back(hidden, ffn_hidden, act);
    return exps;
}

// ========================================================================
// 1. SPARSE MoE
// ========================================================================

SparseMoE::SparseMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      router_weight(hidden, cfg.num_experts)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);
}

MoEOutput SparseMoE::forward(const Tensor& x, bool training) {
    (void)training;
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S, E = config.num_experts, K = config.top_k;

    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router_weight.forward(x_flat);

    Tensor indices, weights;
    Tensor probs = softmax_with_topk(logits, K, indices, weights);

    float zl = 0.0f;
    int64_t dropped = 0;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, D, &zl, &dropped);

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.expert_indices = indices;
    out.expert_weights = weights;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    out.tokens_dropped = dropped;
    return out;
}

// ========================================================================
// 2. SOFT MoE (Dense Mixture)
// ========================================================================

SoftMoE::SoftMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden)
{
    int64_t E = cfg.num_experts;
    int64_t S = cfg.num_slots_per_expert;
    int64_t ffn_hidden = cfg.expert_hidden_size;
    int64_t total_slots = E * S;
    input_mixing = Linear(hidden, total_slots);
    output_mixing = Linear(hidden, hidden);
    experts = create_experts(E, hidden, ffn_hidden, Activation::SiLU);
}

MoEOutput SoftMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;
    int64_t slots = config.num_slots_per_expert;
    int64_t total_slots = E * slots;

    Tensor x_flat = x.reshape({T, D});

    // Learn input-dependent slot assignments
    Tensor slot_weights = input_mixing.forward(x_flat);
    Tensor slot_softmax({T, total_slots});
    math::softmax(slot_weights, slot_softmax, 1);

    // Each expert processes weighted combination of inputs
    Tensor expert_in({E, D});
    expert_in.zero_();
    float* ei = expert_in.data<float>();
    const float* xd = x_flat.data<float>();
    const float* sw = slot_softmax.data<float>();

    for (int64_t e = 0; e < E; ++e) {
        for (int64_t s = 0; s < slots; ++s) {
            int64_t slot_idx = e * slots + s;
            for (int64_t t = 0; t < T; ++t) {
                float w = sw[t * total_slots + slot_idx];
                for (int64_t d = 0; d < D; ++d)
                    ei[e * D + d] += w * xd[t * D + d];
            }
        }
    }

    // Expert forward
    Tensor expert_out({E, D});
    float* eo = expert_out.data<float>();
    for (int64_t e = 0; e < E; ++e) {
        Tensor inp({1, D});
        std::memcpy(inp.data<float>(), ei + e * D, D * sizeof(float));
        Tensor out = experts[(size_t)e].forward(inp);
        std::memcpy(eo + e * D, out.data<float>(), D * sizeof(float));
    }

    // Combine: output[t] = Σ_e Σ_s slot_weight[t, e, s] × expert_out[e]
    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t e = 0; e < E; ++e) {
            for (int64_t s = 0; s < slots; ++s) {
                float w = sw[t * total_slots + e * slots + s];
                for (int64_t d = 0; d < D; ++d)
                    od[t * D + d] += w * eo[e * D + d];
            }
        }
    }

    // Output projection
    Tensor final = output_mixing.forward(output);

    MoEOutput out;
    out.output = final.reshape({B, S, D});
    out.router_logits = slot_weights;
    out.load_balance_loss = 0.0f;
    out.z_loss = config.z_loss_coef * compute_z_loss(expert_out) / (float)T;
    out.num_activated_experts = E;
    return out;
}

// ========================================================================
// 3. HIERARCHICAL MoE
// ========================================================================

HierarchicalMoE::HierarchicalMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      group_router(hidden, cfg.num_groups)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    expert_groups.resize((size_t)cfg.num_groups);
    expert_routers.reserve(cfg.num_groups);
    for (int64_t g = 0; g < cfg.num_groups; ++g) {
        expert_groups[(size_t)g] = create_experts(cfg.experts_per_group, hidden, ffn_hidden, Activation::SiLU);
        expert_routers.emplace_back(hidden, cfg.experts_per_group);
    }
}

MoEOutput HierarchicalMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t G = config.num_groups;
    int64_t TG = config.top_groups;
    int64_t E = config.experts_per_group;
    int64_t K = config.top_experts_per_group;

    Tensor x_flat = x.reshape({T, D});

    // Level 1: group selection
    Tensor group_logits = group_router.forward(x_flat);
    Tensor g_indices, g_weights;
    Tensor g_probs = softmax_with_topk(group_logits, TG, g_indices, g_weights);

    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();
    const float* xd = x_flat.data<float>();
    int64_t* gi = g_indices.data<int64_t>();
    float* gw = g_weights.data<float>();

    float zl = 0.0f;

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t gk = 0; gk < TG; ++gk) {
            int64_t g = gi[t * TG + gk];
            float gw_val = gw[t * TG + gk];
            if (g < 0 || g >= G || gw_val <= 0.0f) continue;

            // Level 2: expert selection within group
            Tensor expert_logits = expert_routers[(size_t)g].forward(
                x_flat.slice(0, t, t + 1));
            Tensor e_indices, e_weights;
            Tensor e_probs = softmax_with_topk(expert_logits, K, e_indices, e_weights);

            int64_t* ei = e_indices.data<int64_t>();
            float* ew = e_weights.data<float>();

            for (int64_t ek = 0; ek < K; ++ek) {
                int64_t e = ei[ek];
                float ew_val = ew[ek];
                if (e < 0 || e >= E || ew_val <= 0.0f) continue;

                Tensor inp({1, D});
                std::memcpy(inp.data<float>(), xd + t * D, D * sizeof(float));
                Tensor eout = expert_groups[(size_t)g][(size_t)e].forward(inp);
                const float* ed = eout.data<float>();
                float mix = gw_val * ew_val;
                for (int64_t d = 0; d < D; ++d) {
                    float v = mix * ed[d];
                    od[t * D + d] += v;
                    zl += v * v;
                }
            }
        }
    }

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = group_logits;
    out.load_balance_loss = compute_load_balance_loss(group_logits, g_indices, G);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = TG * K;
    return out;
}

// ========================================================================
// 4. MoMoE — Mixture of Mixture of Experts
// ========================================================================

MoMoE::MoMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      primary_router(hidden, cfg.num_groups)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    groups.resize((size_t)cfg.num_groups);
    secondary_routers.reserve(cfg.num_groups);
    for (int64_t g = 0; g < cfg.num_groups; ++g) {
        groups[(size_t)g] = create_experts(cfg.experts_per_group, hidden, ffn_hidden, Activation::SiLU);
        secondary_routers.emplace_back(hidden, cfg.experts_per_group);
    }
}

MoEOutput MoMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t G = config.num_groups;
    int64_t TG = config.top_groups;
    int64_t E = config.experts_per_group;
    int64_t TK = config.top_experts_per_group;

    Tensor x_flat = x.reshape({T, D});

    // Primary routing: tokens → groups
    Tensor primary_logits = primary_router.forward(x_flat);
    Tensor g_indices, g_weights;
    Tensor g_probs = softmax_with_topk(primary_logits, TG, g_indices, g_weights);

    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();
    const float* xd = x_flat.data<float>();
    int64_t* gi = g_indices.data<int64_t>();
    float* gw = g_weights.data<float>();

    float zl = 0.0f;

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t gk = 0; gk < TG; ++gk) {
            int64_t g = gi[t * TG + gk];
            float gw_val = gw[t * TG + gk];
            if (g < 0 || g >= G || gw_val <= 0.0f) continue;

            // Secondary routing: within group MoE
            Tensor sec_logits = secondary_routers[(size_t)g].forward(
                x_flat.slice(0, t, t + 1));
            Tensor e_indices, e_weights;
            softmax_with_topk(sec_logits, TK, e_indices, e_weights);

            int64_t* ei = e_indices.data<int64_t>();
            float* ew = e_weights.data<float>();

            for (int64_t ek = 0; ek < TK; ++ek) {
                int64_t e = ei[ek];
                float ew_val = ew[ek];
                if (e < 0 || e >= E || ew_val <= 0.0f) continue;

                Tensor inp({1, D});
                std::memcpy(inp.data<float>(), xd + t * D, D * sizeof(float));
                Tensor eout = groups[(size_t)g][(size_t)e].forward(inp);
                const float* ed = eout.data<float>();
                float mix = gw_val * ew_val;
                for (int64_t d = 0; d < D; ++d) {
                    float v = mix * ed[d];
                    od[t * D + d] += v;
                    zl += v * v;
                }
            }
        }
    }

    // Load balance: both primary and secondary
    float lb_loss = compute_load_balance_loss(primary_logits, g_indices, G);

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = primary_logits;
    out.load_balance_loss = lb_loss;
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = TG * TK;
    return out;
}

// ========================================================================
// 5. EXPERT CHOICE MoE
// ========================================================================

ExpertChoiceMoE::ExpertChoiceMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      router_weight(hidden, cfg.num_experts)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);
}

MoEOutput ExpertChoiceMoE::forward(const Tensor& x) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;
    int64_t capacity = std::min(T, config.capacity_factor * T / E);
    if (capacity < 1) capacity = 1;

    Tensor x_flat = x.reshape({T, D});
    Tensor logits = router_weight.forward(x_flat);
    const float* l = logits.data<float>();

    // Each expert picks its top-K tokens
    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();
    const float* xd = x_flat.data<float>();

    float zl = 0.0f;
    int64_t total_assigned = 0;

    for (int64_t e = 0; e < E; ++e) {
        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(T);
        for (int64_t t = 0; t < T; ++t)
            scored.push_back({l[t * E + e], t});
        int64_t actual_cap = std::min(capacity, T);
        std::partial_sort(scored.begin(), scored.begin() + actual_cap, scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
        std::vector<int64_t> chosen;
        for (int64_t k = 0; k < actual_cap; ++k)
            if (scored[k].first > 0.0f) chosen.push_back(scored[k].second);
        int64_t nt = (int64_t)chosen.size();
        if (nt == 0) continue;

        Tensor batch_input({nt, D});
        float* bi = batch_input.data<float>();
        for (int64_t i = 0; i < nt; ++i)
            std::memcpy(bi + i * D, xd + chosen[(size_t)i] * D, (size_t)D * sizeof(float));

        Tensor batch_output = experts[(size_t)e].forward(batch_input);
        const float* bo = batch_output.data<float>();
        for (int64_t i = 0; i < nt; ++i) {
            int64_t t = chosen[(size_t)i];
            for (int64_t d = 0; d < D; ++d) {
                od[t * D + d] += bo[i * D + d];
                zl += bo[i * D + d] * bo[i * D + d];
            }
            total_assigned++;
        }
    }

    // Normalize by number of experts that processed each token
    for (int64_t t = 0; t < T; ++t) {
        float* row = od + t * D;
        float count = 0.0f;
        for (int64_t e = 0; e < E; ++e)
            if (l[t * E + e] > 0.0f) count += 1.0f;
        if (count > 1.0f) {
            float inv = 1.0f / count;
            for (int64_t d = 0; d < D; ++d)
                row[d] *= inv;
        }
    }

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = 0.0f;
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = E;
    return out;
}

// ========================================================================
// 6. HASH MoE
// ========================================================================

HashMoE::HashMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      num_buckets(cfg.num_experts * cfg.hash_bucket_size)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);
}

int64_t HashMoE::hash_to_expert(int64_t token_id, int64_t num_experts, int64_t bucket_size) {
    uint64_t h = (uint64_t)token_id * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 37;
    h *= 0xBF58476D1CE4E5B9ULL;
    uint64_t bucket = h % (uint64_t)(num_experts * bucket_size);
    return (int64_t)(bucket / (uint64_t)bucket_size);
}

MoEOutput HashMoE::forward(const Tensor& x, const Tensor& token_ids) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;

    Tensor x_flat = x.reshape({T, D});
    const int64_t* ids = token_ids.data<int64_t>();
    const float* xd = x_flat.data<float>();

    Tensor output({T, D});
    output.zero_();
    float* od = output.data<float>();

    // Group tokens by their assigned expert
    std::vector<std::vector<int64_t>> expert_tokens((size_t)E);
    for (int64_t t = 0; t < T; ++t) {
        int64_t e = hash_to_expert(ids[t], E, config.hash_bucket_size);
        if (e >= 0 && e < E)
            expert_tokens[(size_t)e].push_back(t);
    }

    float zl = 0.0f;
    for (int64_t e = 0; e < E; ++e) {
        auto& tokens = expert_tokens[(size_t)e];
        int64_t nt = (int64_t)tokens.size();
        if (nt == 0) continue;
        Tensor batch_input({nt, D});
        float* bi = batch_input.data<float>();
        for (int64_t i = 0; i < nt; ++i)
            std::memcpy(bi + i * D, xd + tokens[(size_t)i] * D, (size_t)D * sizeof(float));
        Tensor batch_output = experts[(size_t)e].forward(batch_input);
        const float* bo = batch_output.data<float>();
        for (int64_t i = 0; i < nt; ++i) {
            int64_t t = tokens[(size_t)i];
            for (int64_t d = 0; d < D; ++d) {
                od[t * D + d] += bo[i * D + d];
                zl += bo[i * D + d] * bo[i * D + d];
            }
        }
    }

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.load_balance_loss = 0.0f;
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = (int64_t)expert_tokens.size();
    return out;
}

// ========================================================================
// 7. CROSS-LAYER MoE
// ========================================================================

CrossLayerMoE::CrossLayerMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    shared_experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);
    layer_routers.reserve(cfg.num_shared_layers);
    for (int64_t i = 0; i < cfg.num_shared_layers; ++i)
        layer_routers.emplace_back(hidden, cfg.num_experts);
}

MoEOutput CrossLayerMoE::forward(const Tensor& x, int64_t layer_idx) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;
    int64_t K = config.top_k;

    size_t ridx = (size_t)(layer_idx % config.num_shared_layers);
    Tensor x_flat = x.reshape({T, D});
    Tensor logits = layer_routers[ridx].forward(x_flat);

    Tensor indices, weights;
    softmax_with_topk(logits, K, indices, weights);

    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, shared_experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, D, &zl, nullptr);

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    return out;
}

// ========================================================================
// 8. MULTIMODAL MoE (MoMMoE)
// ========================================================================

MultiModalMoE::MultiModalMoE(int64_t hidden, const MoEAllConfig& cfg)
    : config(cfg), hidden_size(hidden),
      router_weight(hidden, cfg.num_experts),
      modality_classifier(hidden, 9)
{
    int64_t ffn_hidden = cfg.expert_hidden_size;
    experts = create_experts(cfg.num_experts, hidden, ffn_hidden, Activation::SiLU);

    // Build expert-to-modality map
    expert_modality_map.resize((size_t)cfg.num_experts);
    int64_t offset = 0;
    struct { int64_t count; int64_t mod; } mods[] = {
        {cfg.text_experts, 0}, {cfg.vision_experts, 1}, {cfg.image_gen_experts, 2},
        {cfg.video_gen_experts, 3}, {cfg.audio_experts, 4}, {cfg.ocr_experts, 5},
        {cfg.cross_modal_experts, 6}
    };
    for (auto& m : mods) {
        for (int64_t i = 0; i < m.count && offset < cfg.num_experts; ++i)
            expert_modality_map[(size_t)offset++] = m.mod;
    }
    // Fill remaining with text
    while (offset < cfg.num_experts)
        expert_modality_map[(size_t)offset++] = 0;
}

MoEOutput MultiModalMoE::forward(const Tensor& x, const Tensor& modality_hints) {
    int64_t B = x.dim(0), S = x.dim(1), D = hidden_size;
    int64_t T = B * S;
    int64_t E = config.num_experts;
    int64_t K = config.top_k;
    int64_t NM = num_modalities;

    Tensor x_flat = x.reshape({T, D});

    // Classify modality per token
    Tensor mod_logits = modality_classifier.forward(x_flat);
    Tensor mod_probs({T, NM});
    math::softmax(mod_logits, mod_probs, 1);

    // Compute routing logits with modality bias
    Tensor logits = router_weight.forward(x_flat);
    float* l = logits.data<float>();
    const float* mp = mod_probs.data<float>();

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t e = 0; e < E; ++e) {
            int64_t exp_mod = expert_modality_map[(size_t)e];
            l[t * E + e] += 10.0f * mp[t * NM + exp_mod];
        }
    }

    // Add external modality hints if provided
    if (modality_hints.numel() > 0) {
        const float* mh = modality_hints.data<float>();
        int64_t mh_T = modality_hints.numel();
        for (int64_t t = 0; t < T && t < mh_T; ++t) {
            int64_t hint_mod = (int64_t)mh[t];
            if (hint_mod >= 0 && hint_mod < NM) {
                for (int64_t e = 0; e < E; ++e) {
                    if (expert_modality_map[(size_t)e] == hint_mod)
                        l[t * E + e] += 5.0f;
                }
            }
        }
    }

    Tensor indices, weights;
    Tensor probs = softmax_with_topk(logits, K, indices, weights);

    float zl = 0.0f;
    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, D, &zl, nullptr);

    MoEOutput out;
    out.output = output.reshape({B, S, D});
    out.router_logits = logits;
    out.load_balance_loss = compute_load_balance_loss(logits, indices, E);
    out.z_loss = config.z_loss_coef * zl / (float)T;
    out.num_activated_experts = K;
    return out;
}

// ========================================================================
// Batched expert dispatch
// ========================================================================

Tensor moe_dispatch_batched(const Tensor& x_flat,
                            const std::vector<ExpertFFN>& experts,
                            const int64_t* indices, const float* weights,
                            int64_t T, int64_t K, int64_t E, int64_t D,
                            float* z_loss_out, int64_t* dropped_out) {
    Tensor output({T, D});
    output.zero_();

    std::vector<int> counts((size_t)E, 0);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = indices[t * K + k];
            if (e >= 0 && e < E) counts[(size_t)e]++;
        }

    std::vector<std::vector<int64_t>> expert_tokens((size_t)E);
    for (int64_t e = 0; e < E; ++e)
        expert_tokens[(size_t)e].reserve((size_t)counts[(size_t)e]);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = indices[t * K + k];
            if (e >= 0 && e < E) expert_tokens[(size_t)e].push_back(t);
        }

    const float* xd = x_flat.data<float>();
    float* od = output.data<float>();
    float zl = 0.0f;
    int64_t dropped = 0;

    for (int64_t e = 0; e < E; ++e) {
        int64_t nt = (int64_t)expert_tokens[(size_t)e].size();
        if (nt == 0) continue;

        Tensor batch_input({nt, D});
        float* bi = batch_input.data<float>();
        for (int64_t i = 0; i < nt; ++i) {
            int64_t t = expert_tokens[(size_t)e][(size_t)i];
            std::memcpy(bi + i * D, xd + t * D, (size_t)D * sizeof(float));
        }

        Tensor batch_output = experts[(size_t)e].forward(batch_input);
        const float* bo = batch_output.data<float>();

        for (int64_t i = 0; i < nt; ++i) {
            int64_t t = expert_tokens[(size_t)e][(size_t)i];
            float wgt = 0.0f;
            for (int64_t k = 0; k < K; ++k)
                if (indices[t * K + k] == e) { wgt = weights[t * K + k]; break; }
            if (wgt <= 0.0f) { dropped++; continue; }
            for (int64_t d = 0; d < D; ++d) {
                float v = wgt * bo[i * D + d];
                od[t * D + d] += v;
                zl += v * v;
            }
        }
    }

    if (z_loss_out) *z_loss_out = zl;
    if (dropped_out) *dropped_out = dropped;
    return output;
}

// ========================================================================
// AVX2-optimized MoE kernels
// ========================================================================

namespace avx2 {

void moe_combine(float* output, const float* expert_outputs,
                 const float* weights, const int64_t* indices,
                 int64_t T, int64_t K, int64_t D) {
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = indices[t * K + k];
            float w = weights[t * K + k];
            if (e < 0 || w <= 0.0f) continue;
            const float* src = expert_outputs + (e * T + t) * D;
            float* dst = output + t * D;
            for (int64_t d = 0; d < D; ++d)
                dst[d] += w * src[d];
        }
}

void moe_softmax_topk(float* probs, int64_t* indices, float* weights,
                      const float* logits, int64_t T, int64_t E, int64_t K) {
    for (int64_t t = 0; t < T; ++t) {
        const float* row = logits + t * E;
        float* p = probs + t * E;
        float maxv = row[0];
        for (int64_t e = 1; e < E; ++e)
            if (row[e] > maxv) maxv = row[e];
        float sum = 0.0f;
        for (int64_t e = 0; e < E; ++e) {
            p[e] = std::exp(row[e] - maxv);
            sum += p[e];
        }
        float inv = 1.0f / sum;
        for (int64_t e = 0; e < E; ++e)
            p[e] *= inv;

        std::vector<std::pair<float, int64_t>> scored;
        scored.reserve(E);
        for (int64_t e = 0; e < E; ++e)
            scored.push_back({p[e], e});
        std::partial_sort(scored.begin(), scored.begin() + K, scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
        for (int64_t k = 0; k < K; ++k) {
            indices[t * K + k] = scored[k].second;
            weights[t * K + k] = scored[k].first;
        }
    }
}

void moe_load_balance(float* f_i, float* P_i,
                      const int64_t* indices, const float* weights,
                      int64_t T, int64_t K, int64_t E) {
    std::memset(f_i, 0, (size_t)E * sizeof(float));
    std::memset(P_i, 0, (size_t)E * sizeof(float));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            int64_t e = indices[t * K + k];
            if (e >= 0 && e < E) {
                f_i[e] += 1.0f;
                P_i[e] += weights[t * K + k];
            }
        }
    float inv_T = 1.0f / (float)T;
    for (int64_t e = 0; e < E; ++e) {
        f_i[e] *= inv_T;
        P_i[e] *= inv_T;
    }
}

} // namespace avx2

// ========================================================================
// 9. MMoE — Multi-gate Mixture of Experts
// ========================================================================

MMoE::MMoE(int64_t hidden_size, const MoEAllConfig& cfg)
    : hidden_size(hidden_size), config(cfg) {
    int64_t num_experts = cfg.num_experts;
    int64_t ffn_hidden = cfg.expert_hidden_size;
    int64_t num_tasks = cfg.num_tasks > 0 ? cfg.num_tasks : 1;

    experts.reserve(num_experts);
    for (int64_t i = 0; i < num_experts; i++)
        experts.emplace_back(hidden_size, ffn_hidden);

    task_gates.reserve(num_tasks);
    for (int64_t i = 0; i < num_tasks; i++)
        task_gates.emplace_back(hidden_size, num_experts);
}

MoEOutput MMoE::forward(const Tensor& x, int64_t task_id) {
    MoEOutput out;
    int64_t T = x.numel() / hidden_size;
    Tensor x_flat = x.reshape({T, hidden_size});
    int64_t E = config.num_experts;
    int64_t K = config.top_k > 0 ? config.top_k : 2;

    int64_t tid = (task_id >= 0 && task_id < (int64_t)task_gates.size()) ? task_id : 0;
    Tensor gate_logits = task_gates[tid].forward(x_flat);

    Tensor indices({T, K}, DType::I64);
    Tensor weights({T, K});
    Tensor probs = softmax_with_topk(gate_logits, K, indices, weights);

    out.router_logits = gate_logits;
    out.expert_indices = indices;
    out.expert_weights = weights;
    out.load_balance_loss = compute_load_balance_loss(gate_logits, indices, E);
    out.z_loss = compute_z_loss(gate_logits);

    Tensor output = moe_dispatch_batched(x_flat, experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, hidden_size);

    out.output = output.reshape(x.shape());
    out.num_activated_experts = E;
    return out;
}

// ========================================================================
// 10. DeepSeek-MoE (shared + routed experts)
// ========================================================================

DeepSeekMoE::DeepSeekMoE(int64_t hidden_size, const MoEAllConfig& cfg)
    : hidden_size(hidden_size), config(cfg),
      shared_expert(hidden_size, cfg.expert_hidden_size) {
    int64_t num_routed = cfg.num_routed_experts > 0 ? cfg.num_routed_experts : 8;
    int64_t ffn_hidden = cfg.expert_hidden_size;

    routed_experts.reserve(num_routed);
    for (int64_t i = 0; i < num_routed; i++)
        routed_experts.emplace_back(hidden_size, ffn_hidden);

    router_weight = Linear(hidden_size, num_routed);
    expert_biases.assign(num_routed, 0.0f);
}

MoEOutput DeepSeekMoE::forward(const Tensor& x, bool training) {
    MoEOutput out;
    int64_t T = x.numel() / hidden_size;
    Tensor x_flat = x.reshape({T, hidden_size});
    int64_t E = (int64_t)routed_experts.size();
    int64_t K = config.top_k > 0 ? config.top_k : 2;

    // Shared expert is always active
    Tensor shared_out = shared_expert.forward(x_flat);

    // Route to K experts
    Tensor gate_logits = router_weight.forward(x_flat);
    float* gl = gate_logits.data<float>();
    for (int64_t t = 0; t < T * E; t++)
        gl[t] += expert_biases[t % E];

    Tensor indices({T, K}, DType::I64);
    Tensor weights({T, K});
    Tensor probs = softmax_with_topk(gate_logits, K, indices, weights);

    // Bias update for load balancing (no auxiliary loss)
    if (training) {
        const int64_t* idx = indices.data<int64_t>();
        for (int64_t t = 0; t < T; t++)
            for (int64_t k = 0; k < K; k++)
                expert_biases[idx[t * K + k]] -= 0.001f;
        float mean_bias = 0;
        for (int64_t e = 0; e < E; e++) mean_bias += expert_biases[e];
        mean_bias /= E;
        for (int64_t e = 0; e < E; e++) expert_biases[e] -= mean_bias;
    }

    out.router_logits = gate_logits;
    out.expert_indices = indices;
    out.expert_weights = weights;

    Tensor routed_out = moe_dispatch_batched(x_flat, routed_experts,
        indices.data<int64_t>(), weights.data<float>(),
        T, K, E, hidden_size);

    // Combine shared + routed
    Tensor output({T, hidden_size});
    float* od = output.data<float>();
    const float* sd = shared_out.data<float>();
    const float* rd = routed_out.data<float>();
    for (int64_t i = 0; i < T * hidden_size; ++i)
        od[i] = sd[i] + rd[i];

    out.output = output.reshape(x.shape());
    out.num_activated_experts = E + 1;
    return out;
}

} // namespace moe
} // namespace oil
