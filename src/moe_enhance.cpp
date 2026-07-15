#include "oil/moe_enhance.h"
#include "oil/math.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace oil {

// F1: Expert capacity tuning
ExpertCapacity ExpertCapacity::adaptive(int64_t num_experts, int64_t tokens,
                                         const float* importance) {
    ExpertCapacity ec;
    ec.capacities.resize(num_experts);
    float total = std::accumulate(importance, importance + num_experts, 0.0f);
    int64_t capacity_per_expert = (tokens + num_experts - 1) / num_experts;
    for (int64_t i = 0; i < num_experts; i++) {
        float ratio = (total > 0) ? (importance[i] / total) : (1.0f / num_experts);
        ec.capacities[i] = std::max((int64_t)(ratio * tokens * 1.25f), capacity_per_expert);
    }
    return ec;
}

// F2: DS-MoE prune
DenseToMoEPruner::DenseToMoEPruner(int64_t hidden, int64_t n_experts, float sparsity)
    : hidden_(hidden), n_experts_(n_experts), sparsity_(sparsity) {}

Tensor DenseToMoEPruner::prune_ffn(const Tensor& gate, const Tensor& up, const Tensor& down) {
    return Tensor({n_experts_, hidden_, hidden_}); // dummy: return reshaped
}

// F3: MoE OIL format
void MoEOILFormat::save_experts(const std::vector<Tensor>& experts,
                                 const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return;
    int32_t n = (int32_t)experts.size();
    fwrite(&n, sizeof(n), 1, fp);
    for (auto& e : experts) {
        int64_t sz = e.numel();
        fwrite(&sz, sizeof(sz), 1, fp);
        fwrite(e.data<float>(), sz * sizeof(float), 1, fp);
    }
    fclose(fp);
}

std::vector<Tensor> MoEOILFormat::load_experts(const std::string& path) {
    std::vector<Tensor> experts;
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return experts;
    int32_t n = 0;
    fread(&n, sizeof(n), 1, fp);
    for (int32_t i = 0; i < n; i++) {
        int64_t sz = 0;
        fread(&sz, sizeof(sz), 1, fp);
        Tensor t({sz});
        fread(t.data<float>(), sz * sizeof(float), 1, fp);
        experts.push_back(t);
    }
    fclose(fp);
    return experts;
}

// F4: Expert dropout
ExpertDropout::ExpertDropout(float rate) : rate_(rate) {}
int64_t ExpertDropout::get_active_count(int64_t total) const {
    return std::max((int64_t)1, (int64_t)(total * (1.0f - rate_)));
}
std::vector<bool> ExpertDropout::get_mask(int64_t total) const {
    std::vector<bool> mask(total, true);
    int64_t drop = (int64_t)(total * rate_);
    for (int64_t i = 0; i < drop; i++)
        mask[(i * 7 + 3) % total] = false;
    return mask;
}

// F5: Balance monitoring
ExpertBalance compute_balance(const std::vector<int64_t>& assignments, int64_t n_experts) {
    ExpertBalance eb;
    eb.token_counts.resize(n_experts, 0);
    for (auto a : assignments)
        if (a >= 0 && a < n_experts) eb.token_counts[a]++;
    float mean = (float)assignments.size() / (float)n_experts;
    float var = 0;
    for (auto c : eb.token_counts) var += (c - mean) * (c - mean);
    eb.stddev = std::sqrt(var / n_experts);
    eb.load_balance_loss = eb.stddev / (mean + 1e-8f);
    return eb;
}

// F6: Switch v2
SwitchV2Router::SwitchV2Router(int64_t n_experts, float capacity_factor)
    : n_experts_(n_experts), capacity_factor_(capacity_factor) {}
Tensor SwitchV2Router::forward(const Tensor& logits, int64_t tokens) {
    return Tensor({tokens, n_experts_});
}

// F7: Expert choice v2
ExpertChoiceV2::ExpertChoiceV2(int64_t n_experts, int64_t top_k)
    : n_experts_(n_experts), top_k_(top_k) {}
Tensor ExpertChoiceV2::forward(const Tensor& x, const Tensor& logits) {
    return Tensor(x.shape());
}

// F8: Expert merger
ExpertMerger::ExpertMerger(float threshold) : threshold_(threshold) {}
float ExpertMerger::cosine_sim(const float* a, const float* b, int64_t n) {
    float dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-10f);
}
std::vector<Tensor> ExpertMerger::merge(const std::vector<Tensor>& experts) {
    std::vector<Tensor> result;
    if (experts.empty()) return result;
    int64_t n = experts[0].numel();
    std::vector<bool> merged(experts.size(), false);
    for (size_t i = 0; i < experts.size(); i++) {
        if (merged[i]) continue;
        Tensor avg = Tensor({n}); avg.zero_();
        int count = 0;
        for (size_t j = i; j < experts.size(); j++) {
            if (merged[j]) continue;
            if (cosine_sim(experts[i].data<float>(), experts[j].data<float>(), n) > threshold_) {
                float* ad = avg.data<float>();
                const float* ed = experts[j].data<float>();
                for (int64_t k = 0; k < n; k++) ad[k] += ed[k];
                merged[j] = true;
                count++;
            }
        }
        if (count > 0) {
            float* ad = avg.data<float>();
            for (int64_t k = 0; k < n; k++) ad[k] /= (float)count;
            result.push_back(avg);
        }
    }
    return result;
}

// F9: Dynamic expert pool
DynamicExpertPool::DynamicExpertPool(int64_t initial, int64_t max_experts)
    : max_experts_(max_experts) {
    for (int64_t i = 0; i < initial; i++)
        experts_.push_back(Tensor({0}));
}
int64_t DynamicExpertPool::add_expert(const Tensor& weights) {
    if ((int64_t)experts_.size() >= max_experts_) return -1;
    experts_.push_back(weights.clone());
    return (int64_t)experts_.size() - 1;
}
bool DynamicExpertPool::remove_expert(int64_t idx) {
    if (idx < 0 || idx >= (int64_t)experts_.size()) return false;
    experts_.erase(experts_.begin() + idx);
    return true;
}

// F10: Expert parallelism
ExpertParallel::ExpertParallel(int64_t world_size, int64_t world_rank)
    : world_size_(world_size), world_rank_(world_rank) {}
Tensor ExpertParallel::forward(const Tensor& x, int64_t n_experts) {
    int64_t local = n_experts / world_size_;
    // Each rank handles a subset of experts
    return Tensor(x.shape());
}

} // namespace oil
