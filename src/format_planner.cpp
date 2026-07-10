#include "oil/format_planner.h"
#include <algorithm>
#include <cmath>

namespace oil {

FormatPlanner::FormatPlanner(float t) : target_bpw_(t) {}

void FormatPlanner::score_importance(const Tensor& weights,
                                      const Tensor& calibration_activations,
                                      int block_size) {
    const float* wd = (const float*)weights.data();
    const float* ad = (const float*)calibration_activations.data();
    int64_t n = weights.numel();
    int num_blocks = (int)(n / block_size);
    importance_scores_.resize(num_blocks);

    for (int b = 0; b < num_blocks; b++) {
        float score = 0;
        for (int i = 0; i < block_size && b * block_size + i < n; i++) {
            int idx = b * block_size + i;
            score += std::fabs(wd[idx]) * (ad ? std::fabs(ad[idx]) : 1.0f);
        }
        importance_scores_[b] = score / (float)block_size;
    }
}

FormatPlan FormatPlanner::allocate(int num_weight_blocks, int weights_per_block) {
    FormatPlan plan;
    plan.target_bpw = target_bpw_;
    plan.blocks.resize(num_weight_blocks);
    plan.num_oil8_blocks = 0;
    plan.num_oil4_blocks = 0;
    plan.num_ternary_blocks = 0;
    plan.num_binary_blocks = 0;

    std::vector<int> indices(num_weight_blocks);
    for (int i = 0; i < num_weight_blocks; i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [this](int a, int b) {
        float sa = a < (int)importance_scores_.size() ? importance_scores_[a] : 0;
        float sb = b < (int)importance_scores_.size() ? importance_scores_[b] : 0;
        return sa > sb;
    });

    int oil8_count = (int)(num_weight_blocks * 0.01f);
    int oil4_count = (int)(num_weight_blocks * 0.04f);
    int ternary_count = num_weight_blocks - oil8_count - oil4_count;
    if (ternary_count < 0) {
        oil4_count += ternary_count;
        ternary_count = 0;
    }
    if (oil4_count < 0) {
        oil8_count += oil4_count;
        oil4_count = 0;
    }
    if (oil8_count < 0) oil8_count = 0;

    for (int i = 0; i < num_weight_blocks; i++) {
        int idx = indices[i];
        plan.blocks[idx].id = (uint32_t)idx;
        plan.blocks[idx].weight_index = (uint32_t)(idx * weights_per_block);
        plan.blocks[idx].num_weights = (uint32_t)weights_per_block;
        if (i < oil8_count) {
            plan.blocks[idx].assigned_format = Format::OIL8;
            plan.num_oil8_blocks++;
        } else if (i < oil8_count + oil4_count) {
            plan.blocks[idx].assigned_format = Format::OIL4;
            plan.num_oil4_blocks++;
        } else {
            plan.blocks[idx].assigned_format = Format::TERNARY;
            plan.num_ternary_blocks++;
        }
        if (idx < (int)importance_scores_.size()) {
            plan.blocks[idx].importance_score = importance_scores_[idx];
        } else {
            plan.blocks[idx].importance_score = 0;
        }
    }

    plan.achieved_bpw = estimate_bpw(plan);
    return plan;
}

FormatPlan FormatPlanner::plan_for_model(int64_t num_weights) {
    return allocate((int)(num_weights / 256), 256);
}

const std::vector<float>& FormatPlanner::importance_scores() const {
    return importance_scores_;
}

float FormatPlanner::estimate_bpw(const FormatPlan& plan) {
    if (plan.blocks.empty()) return 0;
    float total = 0;
    for (const auto& b : plan.blocks) {
        total += format_bpw(b.assigned_format);
    }
    return total / (float)plan.blocks.size();
}

void FormatPlanner::sort_by_importance() {
    std::sort(blocks_.begin(), blocks_.end(), [](const WeightBlock& a, const WeightBlock& b) {
        return a.importance_score > b.importance_score;
    });
}

} // namespace oil
