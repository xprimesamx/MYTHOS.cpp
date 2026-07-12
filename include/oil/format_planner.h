#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/codebook.h"
#include <vector>
#include <string>

namespace oil {

struct WeightBlock {
    uint32_t id;
    uint32_t weight_index; // start index in weight tensor
    uint32_t num_weights;
    Format assigned_format;
    float importance_score;
};

struct FormatPlan {
    float target_bpw;
    float achieved_bpw;
    std::vector<WeightBlock> blocks;
    
    // Breakdown
    int num_oil8_blocks;
    int num_oil4_blocks;
    int num_ternary_blocks;
    int num_binary_blocks;
};

enum class ImportanceMetric {
    MAGNITUDE,     // |weight| * |activation|
    FISHER_DIAG,   // gradient^2 (Fisher diagonal)
    HESSIAN_DIAG,  // second-order diagonal
};

class FormatPlanner {
public:
    FormatPlanner(float target_bpw = 1.50f);
    
    void score_importance(const Tensor& weights,
                          const Tensor& calibration_activations,
                          int block_size = 256,
                          ImportanceMetric metric = ImportanceMetric::MAGNITUDE);
    
    void score_importance_fisher(const Tensor& weights,
                                 const Tensor& gradients,
                                 int block_size = 256);
    
    FormatPlan allocate(int num_weight_blocks, int weights_per_block = 256);
    
    FormatPlan plan_for_model(int64_t num_weights);
    
    const std::vector<float>& importance_scores() const;
    
    void set_target_bpw(float bpw) { target_bpw_ = bpw; }
    float target_bpw() const { return target_bpw_; }
    
    static float estimate_bpw(const FormatPlan& plan);
    
    // Find optimal format mix for a target BPW
    static void compute_format_mix(int num_blocks, float target_bpw,
                                   int& oil8, int& oil4, int& ternary, int& binary);
    
private:
    float target_bpw_;
    std::vector<float> importance_scores_;
};

} // namespace oil
