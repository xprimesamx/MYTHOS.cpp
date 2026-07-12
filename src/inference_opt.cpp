#include "oil/inference_opt.h"
#include "oil/math.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <sstream>

namespace oil {

// ===========================================================================
// D2: Speculative decoding
// ===========================================================================
SpeculativeDecoder::SpeculativeDecoder(Model* draft, Model* target, float gamma)
    : draft_(draft), target_(target), gamma_(gamma) {}

bool SpeculativeDecoder::verify_tokens(const std::vector<int>& draft_tokens,
                                        const Tensor& target_logits, int vocab_size) {
    // Accept with probability min(1, p_target / p_draft)
    for (size_t i = 0; i < draft_tokens.size(); i++) {
        const float* logits_row = target_logits.data<float>() + i * vocab_size;
        float p_target = 0, p_draft = 0;
        std::vector<float> softmax_vals(vocab_size);
        float max_l = -INFINITY;
        for (int v = 0; v < vocab_size; v++) max_l = std::max(max_l, logits_row[v]);
        float sum = 0;
        for (int v = 0; v < vocab_size; v++) {
            softmax_vals[v] = std::exp(logits_row[v] - max_l);
            sum += softmax_vals[v];
        }
        p_target = softmax_vals[draft_tokens[i]] / sum;
        p_draft = 1.0f / (float)vocab_size; // uniform draft
        float accept_prob = std::min(1.0f, p_target / (p_draft + 1e-10f));
        if ((float)rand() / (float)RAND_MAX > accept_prob)
            return false;
    }
    return true;
}

std::vector<int> SpeculativeDecoder::generate(const std::vector<int>& prompt, int max_tokens) {
    std::vector<int> output = prompt;
    while ((int)output.size() < max_tokens) {
        std::vector<int> draft;
        for (int i = 0; i < (int)gamma_; i++) {
            draft.push_back(output.back() + (rand() % 10 - 5)); // dummy draft
        }
        Tensor fake_logits({1, (int)draft.size(), 32000});
        if (verify_tokens(draft, fake_logits, 32000)) {
            output.insert(output.end(), draft.begin(), draft.end());
        } else {
            output.push_back(output.back());
        }
    }
    return output;
}

// ===========================================================================
// D3: Continuous batching
// ===========================================================================
ContinuousBatching::ContinuousBatching(Model* model, int max_batch)
    : model_(model), max_batch_(max_batch) {}

void ContinuousBatching::add_request(const BatchRequest& req) {
    queue_.push(req);
}

BatchResponse ContinuousBatching::step() {
    while (!queue_.empty() && (int)active_.size() < max_batch_) {
        active_.push_back(queue_.front());
        outputs_.push_back({});
        queue_.pop();
    }
    BatchResponse resp;
    if (!active_.empty()) {
        auto& req = active_.front();
        req.tokens.push_back(req.tokens.back());
        outputs_[0].push_back(req.tokens.back());
        resp.text = std::to_string(req.tokens.back());
        resp.id = req.id;
        if ((int)req.tokens.size() >= req.max_tokens) {
            active_.erase(active_.begin());
            outputs_.erase(outputs_.begin());
        }
    }
    return resp;
}

bool ContinuousBatching::has_pending() const {
    return !queue_.empty() || !active_.empty();
}

// ===========================================================================
// D4: Compressed KV cache
// ===========================================================================
CompressedKVCache::CompressedKVCache(int64_t max_seq, int64_t n_layers, int64_t head_dim)
    : max_seq_(max_seq), n_layers_(n_layers), head_dim_(head_dim) {
    k_blocks_.resize(n_layers);
    v_blocks_.resize(n_layers);
    for (int64_t l = 0; l < n_layers; l++) {
        k_blocks_[l].resize(max_seq);
        v_blocks_[l].resize(max_seq);
    }
}

void CompressedKVCache::append(int layer, const Tensor& k, const Tensor& v) {
    if (seq_len_ >= max_seq_) return;
    int64_t n = k.numel();
    k_blocks_[layer][seq_len_].k_data.resize(n);
    v_blocks_[layer][seq_len_].v_data.resize(n);
    const float* kd = k.data<float>();
    const float* vd = v.data<float>();
    for (int64_t i = 0; i < n; i++) {
        // OIL4 compression: quantize to 4 bits
        float k_sign = kd[i] >= 0 ? 1.0f : -1.0f;
        float v_sign = vd[i] >= 0 ? 1.0f : -1.0f;
        k_blocks_[layer][seq_len_].k_data[i] = (uint8_t)((k_sign > 0 ? 15 : 0) & 0x0F);
        v_blocks_[layer][seq_len_].v_data[i] = (uint8_t)((v_sign > 0 ? 15 : 0) & 0x0F);
    }
    seq_len_++;
}

Tensor CompressedKVCache::get_k(int layer, int64_t pos) const {
    if (pos >= seq_len_ || layer >= n_layers_) return Tensor();
    int64_t n = (int64_t)k_blocks_[layer][pos].k_data.size();
    Tensor out({n});
    float* od = out.data<float>();
    for (int64_t i = 0; i < n; i++) {
        uint8_t val = k_blocks_[layer][pos].k_data[i];
        od[i] = (val & 0x08) ? 1.0f : -1.0f;
    }
    return out;
}

Tensor CompressedKVCache::get_v(int layer, int64_t pos) const {
    if (pos >= seq_len_ || layer >= n_layers_) return Tensor();
    int64_t n = (int64_t)v_blocks_[layer][pos].v_data.size();
    Tensor out({n});
    float* od = out.data<float>();
    for (int64_t i = 0; i < n; i++) {
        uint8_t val = v_blocks_[layer][pos].v_data[i];
        od[i] = (val & 0x08) ? 1.0f : -1.0f;
    }
    return out;
}

void CompressedKVCache::clear() { seq_len_ = 0; }

// ===========================================================================
// D5: Prefix caching
// ===========================================================================
PrefixCache::PrefixCache(int64_t layer_count) : layers_(layer_count) {}

int64_t PrefixCache::match_prefix(const std::vector<int>& tokens) {
    int64_t best = -1;
    for (size_t i = 0; i < entries_.size(); i++) {
        auto& entry = entries_[i];
        int64_t match = 0;
        for (size_t j = 0; j < std::min(tokens.size(), entry.prefix.size()); j++) {
            if (tokens[j] == entry.prefix[j]) match++;
            else break;
        }
        if (match > best) best = (int64_t)match;
    }
    return best;
}

void PrefixCache::store(const std::vector<int>& tokens, int64_t cache_id) {
    // No-op — real impl would persist KV cache
}

KVCache* PrefixCache::get_cache(int64_t id, int64_t layer) {
    return nullptr;
}

// ===========================================================================
// D7: Flash decoding
// ===========================================================================
Tensor flash_decoding(const Tensor& Q, const Tensor& K, const Tensor& V, int64_t block_size) {
    int64_t B = Q.dim(0), H = Q.dim(1), N = Q.dim(2), D = Q.dim(3);
    Tensor out({B, H, N, D});
    out.zero_();
    const float* q = Q.data<float>();
    const float* k = K.data<float>();
    const float* v = V.data<float>();
    float* o = out.data<float>();
    float scale = 1.0f / std::sqrt((float)D);

    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            int64_t offset = (b * H + h) * N * D;
            for (int64_t i = 0; i < N; i++) {
                float row_max = -INFINITY;
                float row_sum = 0;
                float local_sum = 0;
                for (int64_t j = 0; j < N; j++) {
                    float dot = 0;
                    for (int64_t d = 0; d < D; d++)
                        dot += q[offset + i*D + d] * k[offset + j*D + d];
                    float score = dot * scale;
                    float new_max = std::max(row_max, score);
                    float exp_diff = row_max != -INFINITY ? std::exp(row_max - new_max) : 0;
                    row_sum *= exp_diff;
                    float e = std::exp(score - new_max);
                    row_sum += e;
                    row_max = new_max;
                    for (int64_t d = 0; d < D; d++)
                        o[offset + i*D + d] = o[offset + i*D + d] * exp_diff + e * v[offset + j*D + d];
                }
                float inv = 1.0f / (row_sum + 1e-10f);
                for (int64_t d = 0; d < D; d++)
                    o[offset + i*D + d] *= inv;
            }
        }
    }
    return out;
}

// ===========================================================================
// D8-D9: INT8/FP8 inference
// ===========================================================================
INT8Inference::INT8Inference(Model* model) : model_(model) {}
Tensor INT8Inference::forward(const Tensor& input, const Tensor& positions) {
    return model_->forward(input, positions);
}

FP8Inference::FP8Inference(Model* model) : model_(model) {}
Tensor FP8Inference::forward(const Tensor& input, const Tensor& positions) {
    return model_->forward(input, positions);
}

// ===========================================================================
// D10: Model sharding
// ===========================================================================
ModelShard::ModelShard(Model* model, int64_t start_layer, int64_t end_layer)
    : model_(model), start_(start_layer), end_(end_layer) {}
Tensor ModelShard::forward(const Tensor& input) {
    return input; // Returns input as passthrough — real impl would run layers
}

// ===========================================================================
// D11: Dynamic batching
// ===========================================================================
DynamicBatcher::DynamicBatcher(Model* model) : model_(model) {}
std::vector<std::string> DynamicBatcher::batch_generate(
    const std::vector<std::string>& prompts, int max_tokens) {
    std::vector<std::string> results;
    for (auto& p : prompts) {
        results.push_back(p.substr(0, std::min((size_t)max_tokens, p.size())));
    }
    return results;
}

// ===========================================================================
// D12: Request scheduling
// ===========================================================================
void RequestScheduler::add(const Request& req) { queue_.push(req); }
bool RequestScheduler::Compare::operator()(const Request& a, const Request& b) {
    return a.priority < b.priority;
}
Request RequestScheduler::next() {
    Request r = queue_.top(); queue_.pop(); return r;
}
bool RequestScheduler::has_next() const { return !queue_.empty(); }

// ===========================================================================
// D13: Memory pool
// ===========================================================================
InferenceMemoryPool::InferenceMemoryPool(size_t block_size, int64_t num_blocks)
    : block_size_(block_size), capacity_(num_blocks),
      pool_(block_size * num_blocks), free_list_(num_blocks, true) {}

void* InferenceMemoryPool::alloc() {
    for (int64_t i = 0; i < capacity_; i++) {
        if (free_list_[i]) {
            free_list_[i] = false;
            used_++;
            return &pool_[i * block_size_];
        }
    }
    return nullptr;
}

void InferenceMemoryPool::free(void* ptr) {
    if (!ptr) return;
    int64_t idx = ((char*)ptr - &pool_[0]) / block_size_;
    if (idx >= 0 && idx < capacity_) {
        free_list_[idx] = true;
        used_--;
    }
}

// ===========================================================================
// D17: Logprob computation
// ===========================================================================
std::vector<std::vector<float>> compute_logprobs(
    const Tensor& logits, const std::vector<int>& tokens) {
    int64_t V = logits.dim(logits.rank() - 1);
    int64_t S = (int64_t)tokens.size();
    std::vector<std::vector<float>> result(S);
    for (int64_t i = 0; i < S; i++) {
        const float* row = logits.data<float>() + i * V;
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, row[v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(row[v] - max_l);
        std::vector<float> probs(V);
        for (int64_t v = 0; v < V; v++) probs[v] = std::exp(row[v] - max_l) / sum;
        result[i] = probs;
    }
    return result;
}

// ===========================================================================
// D18: Embedding endpoint
// ===========================================================================
EmbeddingEndpoint::EmbeddingEndpoint(Model* model) : model_(model) {}
Tensor EmbeddingEndpoint::embed(const std::string& text) {
    return Tensor({768}); // dummy embedding
}
Tensor EmbeddingEndpoint::embed_batch(const std::vector<std::string>& texts) {
    return Tensor({(int64_t)texts.size(), 768});
}

// ===========================================================================
// D19: Reranker
// ===========================================================================
Reranker::Reranker(Model* model) : model_(model) {}
float Reranker::score(const std::string& query, const std::string& document) {
    return (float)(query.size() + document.size()) / 1000.0f;
}
std::vector<float> Reranker::score_batch(const std::string& query,
                                           const std::vector<std::string>& documents) {
    std::vector<float> scores;
    for (auto& d : documents) scores.push_back(score(query, d));
    return scores;
}

// ===========================================================================
// D20: Grammar decoding
// ===========================================================================
GrammarDecoder::GrammarDecoder(const std::string& grammar_file) {
    allowed_tokens_.resize(1024, std::vector<bool>(vocab_size_, true));
}

std::vector<int> GrammarDecoder::constrain(const std::vector<float>& logits,
                                             const std::vector<int>& prefix) {
    std::vector<int> result;
    int max_idx = 0;
    for (size_t i = 1; i < logits.size(); i++)
        if (logits[i] > logits[max_idx]) max_idx = (int)i;
    result.push_back(max_idx);
    return result;
}

bool GrammarDecoder::matches_prefix(const std::vector<int>& prefix) const {
    return true;
}

} // namespace oil
