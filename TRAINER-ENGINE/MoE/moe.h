#pragma once
#include <vector>
#include <cstdint>
#include "oil/tensor.h"
#include "oil/transformer.h"

namespace oil {
namespace moe {

struct MoEConfig {
    int64_t num_experts = 8;
    int64_t top_k = 2;
    int64_t expert_hidden_size = 2048;
    float load_balance_coef = 0.01f;
    float z_loss_coef = 0.001f;
};

struct RouterOutput {
    Tensor gates;
    Tensor indices;
    Tensor weights;
    float load_balance_loss;
};

class MoERouter {
public:
    explicit MoERouter(int64_t hidden_size, const MoEConfig& cfg);
    RouterOutput forward(const Tensor& x);
    Tensor gate_weight;
private:
    MoEConfig config_;
};

class MoEFFN {
public:
    explicit MoEFFN(const MoEConfig& cfg, const TransformerConfig& tcfg);
    Tensor forward(const Tensor& x, const RouterOutput& routing);
    std::vector<FFN> experts;
private:
    MoEConfig config_;
};

class MoEBlock {
public:
    MoEBlock(const MoEConfig& moe_cfg, const TransformerConfig& tcfg);
    Tensor forward(const Tensor& x, const Tensor& positions,
                   const Tensor& mask, KVCache& cache, int layer_idx);
    RMSNorm attention_norm;
    Attention attention;
    RMSNorm moe_norm;
    MoERouter router;
    MoEFFN moe_ffn;
};

} // namespace moe
} // namespace oil
