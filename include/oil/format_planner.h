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

class FormatPlanner {
public:
    FormatPlanner(float target_bpw = 1.50f);
    
    // Score weight blocks by importance using activation magnitudes
    void score_importance(const Tensor& weights, 
                          const Tensor& calibration_activations,
                          int block_size = 256);
    
    // Allocate formats to meet BPW target
    FormatPlan allocate(int num_weight_blocks, int weights_per_block = 256);
    
    // Plan for a specific model size
    FormatPlan plan_for_model(int64_t num_weights);
    
    const std::vector<float>& importance_scores() const;
    
    // Utility: estimate total BPW from a plan
    static float estimate_bpw(const FormatPlan& plan);
    
private:
    float target_bpw_;
    std::vector<float> importance_scores_;
    std::vector<WeightBlock> blocks_;
    
    void sort_by_importance();
};

} // namespace oil
