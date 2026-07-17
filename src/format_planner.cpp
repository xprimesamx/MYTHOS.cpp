#include "oil/format_planner.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>

namespace oil {

FormatPlanner::FormatPlanner(float t) : target_bpw_(t) {}

void FormatPlanner::score_importance(const Tensor& weights,
                                      const Tensor& calibration_activations,
                                      int block_size, ImportanceMetric metric) {
    if (metric == ImportanceMetric::FISHER_DIAG) return;
    const float* wd = (const float*)weights.data();
    const float* ad = (const float*)calibration_activations.data();
    int64_t n = weights.numel();
    int64_t num_blocks = n / block_size;
    if (n % block_size != 0) num_blocks++;
    importance_scores_.resize((size_t)num_blocks);

    for (int64_t b = 0; b < num_blocks; b++) {
        double score = 0;
        int64_t start = b * block_size;
        int64_t end = (std::min)(start + block_size, n);
        for (int64_t i = start; i < end; i++) {
            float w = wd[i];
            float a = ad ? std::fabs(ad[i]) : 1.0f;
            score += (double)std::fabs(w) * (double)a;
        }
        importance_scores_[b] = (float)(score / (double)(end - start));
    }
}

void FormatPlanner::score_importance_fisher(const Tensor& weights,
                                             const Tensor& gradients,
                                             int block_size) {
    const float* wd = (const float*)weights.data();
    const float* gd = (const float*)gradients.data();
    int64_t n = weights.numel();
    int64_t num_blocks = n / block_size;
    if (n % block_size != 0) num_blocks++;
    importance_scores_.resize((size_t)num_blocks);

    for (int64_t b = 0; b < num_blocks; b++) {
        double score = 0;
        int64_t start = b * block_size;
        int64_t end = (std::min)(start + block_size, n);
        for (int64_t i = start; i < end; i++) {
            float g = gd[i];
            score += (double)(g * g);
        }
        score = std::sqrt(score / (double)(end - start));
        score *= (double)std::fabs(wd[start]);
        importance_scores_[b] = (float)score;
    }
}

void FormatPlanner::compute_format_mix(int num_blocks, float target_bpw,
                                        int& oil8, int& oil4,
                                        int& ternary, int& binary) {
    const float bpw_oil8 = 8.0f;
    const float bpw_oil4 = 4.0f; (void)bpw_oil4;
    const float bpw_ternary = 1.58f; (void)bpw_ternary;
    const float bpw_binary = 1.0f;

    if (target_bpw >= bpw_oil8) {
        oil8 = num_blocks;
        oil4 = 0; ternary = 0; binary = 0;
        return;
    }

    if (target_bpw <= bpw_binary) {
        binary = num_blocks;
        oil8 = 0; oil4 = 0; ternary = 0;
        return;
    }

    oil8 = 0; oil4 = 0; ternary = 0; binary = 0;

    for (int trial_oil8 = 0; trial_oil8 <= num_blocks; trial_oil8++) {
        int remaining = num_blocks - trial_oil8;
        for (int trial_oil4 = 0; trial_oil4 <= remaining; trial_oil4++) {
            int rem2 = remaining - trial_oil4;
            for (int trial_ternary = 0; trial_ternary <= rem2; trial_ternary++) {
                int trial_binary = rem2 - trial_ternary;
                float achieved = (float)(trial_oil8 * 8 + trial_oil4 * 4 +
                                         trial_ternary * 158 + trial_binary * 100) / 100.0f / (float)num_blocks;
                if (achieved >= target_bpw - 0.05f && achieved <= target_bpw + 0.05f) {
                    oil8 = trial_oil8;
                    oil4 = trial_oil4;
                    ternary = trial_ternary;
                    binary = trial_binary;
                    return;
                }
            }
        }
    }

    oil4 = (int)((float)num_blocks * (target_bpw - 1.58f) / (4.0f - 1.58f) + 0.5f);
    oil4 = (std::max)(0, (std::min)(num_blocks, oil4));
    ternary = num_blocks - oil4;
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

    if (!importance_scores_.empty()) {
        std::sort(indices.begin(), indices.end(), [this](int a, int b) {
            float sa = a < (int)importance_scores_.size() ? importance_scores_[a] : 0;
            float sb = b < (int)importance_scores_.size() ? importance_scores_[b] : 0;
            return sa > sb;
        });
    }

    int oil8_count = 0, oil4_count = 0, ternary_count = 0, binary_count = 0;
    compute_format_mix(num_weight_blocks, target_bpw_,
                       oil8_count, oil4_count, ternary_count, binary_count);

    plan.num_oil8_blocks = oil8_count;
    plan.num_oil4_blocks = oil4_count;
    plan.num_ternary_blocks = ternary_count;
    plan.num_binary_blocks = binary_count;

    for (int i = 0; i < num_weight_blocks; i++) {
        int idx = indices[i];
        plan.blocks[idx].id = (uint32_t)idx;
        plan.blocks[idx].weight_index = (uint32_t)(idx * weights_per_block);
        plan.blocks[idx].num_weights = (uint32_t)weights_per_block;
        if (i < oil8_count) {
            plan.blocks[idx].assigned_format = Format::OIL8;
        } else if (i < oil8_count + oil4_count) {
            plan.blocks[idx].assigned_format = Format::OIL4;
        } else if (i < oil8_count + oil4_count + ternary_count) {
            plan.blocks[idx].assigned_format = Format::TERNARY;
        } else {
            plan.blocks[idx].assigned_format = Format::BINARY;
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

} // namespace oil
