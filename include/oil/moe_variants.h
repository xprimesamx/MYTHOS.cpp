#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>
#include "oil/tensor.h"
#include "oil/transformer.h"
#include "oil/math.h"
#include "oil/kv_cache.h"

namespace oil {
namespace moe {

// ========================================================================
// Enums for MoE variant types and routing strategies
// ========================================================================

enum class MoEVariant {
    SPARSE_TOP1 = 0,
    SPARSE_TOP2,
    SPARSE_TOPK,
    SOFT_MIXTURE,
    HIERARCHICAL,
    MOMOE,              // Mixture of Mixture of Experts
    EXPERT_CHOICE,
    HASH_ROUTED,
    CROSS_LAYER,
    MULTIMODAL           // Our MoMMoE
};

inline const char* moe_variant_name(MoEVariant v) {
    switch (v) {
        case MoEVariant::SPARSE_TOP1: return "SPARSE_TOP1";
        case MoEVariant::SPARSE_TOP2: return "SPARSE_TOP2";
        case MoEVariant::SPARSE_TOPK: return "SPARSE_TOPK";
        case MoEVariant::SOFT_MIXTURE: return "SOFT_MIXTURE";
        case MoEVariant::HIERARCHICAL: return "HIERARCHICAL";
        case MoEVariant::MOMOE: return "MOMOE";
        case MoEVariant::EXPERT_CHOICE: return "EXPERT_CHOICE";
        case MoEVariant::HASH_ROUTED: return "HASH_ROUTED";
        case MoEVariant::CROSS_LAYER: return "CROSS_LAYER";
        case MoEVariant::MULTIMODAL: return "MULTIMODAL";
        default: return "UNKNOWN";
    }
}

enum class CapacityStrategy {
    FIXED,      // Fixed capacity per expert
    TOKEN_DROP, // Overflow tokens dropped
    TOKEN_PASS  // Overflow passed via residual
};

// ========================================================================
// MoE Config (extended for all variants)
// ========================================================================

struct MoEAllConfig {
    // General
    MoEVariant variant = MoEVariant::SPARSE_TOPK;
    int64_t num_experts = 16;
    int64_t top_k = 2;
    int64_t expert_hidden_size = 2048;
    float load_balance_coef = 0.01f;
    float z_loss_coef = 0.001f;

    // Hierarchical / MoMoE
    int64_t num_groups = 4;
    int64_t experts_per_group = 4;
    int64_t top_groups = 2;
    int64_t top_experts_per_group = 1;
    float group_load_balance_coef = 0.005f;

    // Soft MoE
    int64_t num_slots_per_expert = 1;

    // Expert Choice
    int64_t capacity_factor = 2;

    // Hash MoE
    int64_t hash_bucket_size = 2;

    // Cross-layer
    int64_t num_shared_layers = 4;

    // Capacity
    CapacityStrategy capacity_strategy = CapacityStrategy::TOKEN_DROP;

    // Modality (for MULTIMODAL variant)
    int64_t text_experts = 4;
    int64_t vision_experts = 3;
    int64_t image_gen_experts = 2;
    int64_t video_gen_experts = 2;
    int64_t audio_experts = 2;
    int64_t ocr_experts = 1;
    int64_t cross_modal_experts = 2;
};

// ========================================================================
// Router output (unified for all variants)
// ========================================================================

struct MoEOutput {
    Tensor output;           // {B, S, hidden}
    Tensor router_logits;    // {T, E} or {T, G} for hierarchical
    Tensor expert_weights;   // {T, K} or variable
    Tensor expert_indices;   // {T, K} or variable
    float load_balance_loss = 0.0f;
    float z_loss = 0.0f;
    int64_t num_activated_experts = 0;
    int64_t tokens_dropped = 0;
};

// ========================================================================
// Expert FFN (shared by all variants)
// ========================================================================

class ExpertFFN {
public:
    Linear gate_proj, up_proj, down_proj;
    Activation activation;

    ExpertFFN() : activation(Activation::SiLU) {}
    explicit ExpertFFN(int64_t hidden_size, int64_t ffn_hidden, Activation act = Activation::SiLU)
        : gate_proj(hidden_size, ffn_hidden),
          up_proj(hidden_size, ffn_hidden),
          down_proj(ffn_hidden, hidden_size),
          activation(act) {}

    Tensor forward(const Tensor& x) const {
        Tensor gate = gate_proj.forward(x);
        Tensor up = up_proj.forward(x);
        Tensor act_out({gate.shape()});
        if (activation == Activation::SiLU) {
            math::silu(gate, act_out);
        } else if (activation == Activation::GELU) {
            math::gelu(gate, act_out);
        } else {
            math::relu(gate, act_out);
        }
        Tensor gated({gate.shape()});
        math::mul(act_out, up, gated);
        return down_proj.forward(gated);
    }
};

// ========================================================================
// 1. SPARSE MoE (Top-1, Top-2, Top-K Token Choice)
// ========================================================================

class SparseMoE {
public:
    SparseMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, bool training = true);

    std::vector<ExpertFFN> experts;
    Linear router_weight;
    MoEAllConfig config;
    int64_t hidden_size;

private:
    MoEOutput route_topk(const Tensor& x, int K);
};

// ========================================================================
// 2. SOFT MoE (Dense Mixture — all experts, learned mixing weights)
// ========================================================================
// From: "From Sparse to Soft Mixtures of Experts" (Puigcerver et al., ICLR 2024)
// Each expert processes a learned weighted combination of ALL input tokens.
// Output = weighted combination of expert outputs.

class SoftMoE {
public:
    SoftMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);

    std::vector<ExpertFFN> experts;
    Linear input_mixing;     // {hidden, num_slots * num_experts}
    Linear output_mixing;    // {num_slots * num_experts, hidden}
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 3. HIERARCHICAL MoE (Tree Gating — Jordan & Jacobs, 1994)
// ========================================================================
// First-level router selects expert GROUPS, second-level selects experts
// within the selected group. Tree structure: root → interior → leaf.

class HierarchicalMoE {
public:
    HierarchicalMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);

    std::vector<std::vector<ExpertFFN>> expert_groups;  // [group][expert]
    Linear group_router;
    std::vector<Linear> expert_routers;  // per group
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 4. MoMoE — Mixture of Mixture of Experts (2-level hierarchical)
// ========================================================================
// Level 1: routes tokens to top-K expert groups
// Level 2: within each selected group, routes to top-K experts
// Output = weighted combination of (group_weight × expert_weight)

class MoMoE {
public:
    MoMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);

    std::vector<std::vector<ExpertFFN>> groups;  // [group][expert]
    Linear primary_router;    // {hidden, num_groups}
    std::vector<Linear> secondary_routers;  // per group: {hidden, experts_per_group}
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 5. EXPERT CHOICE MoE (Zhou et al., 2022)
// ========================================================================
// Instead of tokens picking experts, EACH EXPERT picks its top-K tokens.
// Perfect load balance by construction.

class ExpertChoiceMoE {
public:
    ExpertChoiceMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x);

    std::vector<ExpertFFN> experts;
    Linear router_weight;     // {hidden, num_experts}
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 6. HASH MoE (Roller et al., 2021 — deterministic routing)
// ========================================================================
// No learned router. Expert assignment = hash(token_id) mod num_experts.
// Simple, no load balancing loss, deterministic.

class HashMoE {
public:
    HashMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, const Tensor& token_ids);

    std::vector<ExpertFFN> experts;
    MoEAllConfig config;
    int64_t hidden_size;
    int64_t num_buckets;

    static int64_t hash_to_expert(int64_t token_id, int64_t num_experts, int64_t bucket_size);
};

// ========================================================================
// 7. CROSS-LAYER MoE (Shared experts across transformer layers)
// ========================================================================
// Same expert pool shared by multiple transformer layers.
// Each layer can route to different experts from the shared pool.

class CrossLayerMoE {
public:
    CrossLayerMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, int64_t layer_idx);

    std::vector<ExpertFFN> shared_experts;
    std::vector<Linear> layer_routers;  // one per shared layer
    MoEAllConfig config;
    int64_t hidden_size;
};

// ========================================================================
// 8. MULTIMODAL MoE (MoMMoE — Modality-Aware Sparse MoE)
// ========================================================================
// Extension of SparseMoE with modality-aware routing bias.
// Already partially in engines/trainer/moe/moe.h, here as standalone.

class MultiModalMoE {
public:
    MultiModalMoE(int64_t hidden_size, const MoEAllConfig& cfg);
    MoEOutput forward(const Tensor& x, const Tensor& modality_hints);

    std::vector<ExpertFFN> experts;
    Linear router_weight;
    Linear modality_classifier;
    MoEAllConfig config;
    int64_t hidden_size;
    int64_t num_modalities = 9;

    std::vector<int64_t> expert_modality_map;  // which modality each expert prefers
};

// ========================================================================
// Load balancing utility (shared by all variants)
// ========================================================================

float compute_load_balance_loss(const Tensor& router_logits, const Tensor& expert_indices, int64_t num_experts);
float compute_z_loss(const Tensor& expert_output_norms);
Tensor softmax_with_topk(const Tensor& logits, int64_t k, Tensor& indices_out, Tensor& weights_out);
int64_t hash_token(int64_t token_id, int64_t range);

// ========================================================================
// AVX2-optimized MoE kernels
// ========================================================================

namespace avx2 {
    void moe_combine(float* output, const float* expert_outputs,
                     const float* weights, const int64_t* indices,
                     int64_t T, int64_t K, int64_t D);
    void moe_softmax_topk(float* probs, int64_t* indices, float* weights,
                          const float* logits, int64_t T, int64_t E, int64_t K);
    void moe_load_balance(float* f_i, float* P_i,
                          const int64_t* indices, const float* weights,
                          int64_t T, int64_t K, int64_t E);
}

} // namespace moe
} // namespace oil
