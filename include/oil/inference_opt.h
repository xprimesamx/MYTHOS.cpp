#pragma once
#include "oil/tensor.h"
#include "oil/model.h"
#include "oil/kv_cache.h"
#include "oil/generator.h"
#include "oil/sampler.h"
#include <queue>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <mutex>

namespace oil {

// D1: Paged attention — vLLM-style block-level KV cache management
class PagedAttention {
public:
    struct Block { int64_t id; Tensor k; Tensor v; bool active; };
    PagedAttention(int64_t head_dim, int64_t n_heads, int64_t block_size = 32,
                   int64_t max_blocks = 256);
    Tensor forward(const Tensor& Q, int64_t* block_table,
                   Block* blocks, int64_t num_blocks);
    Block alloc_block();
    void free_block(int64_t id);
    int64_t num_blocks() const { return (int64_t)blocks_.size(); }
    int64_t available_blocks() const { return (int64_t)free_ids_.size(); }
private:
    int64_t head_dim_, n_heads_, block_size_;
    std::vector<Block> blocks_;
    std::queue<int64_t> free_ids_;
    int64_t max_blocks_;
    int64_t next_id_;
};

// D2: Speculative decoding — draft model + verify + adaptive gamma
class SpeculativeDecoder {
public:
    SpeculativeDecoder(Model* draft, Model* target, float gamma = 5.0f,
                       float min_gamma = 1.0f, float max_gamma = 10.0f);
    std::vector<int> generate(const std::vector<int>& prompt, int max_tokens);
    float acceptance_rate() const { return acceptance_rate_; }
    int accepted_count() const { return accepted_count_; }
    int total_count() const { return total_count_; }
    float current_gamma() const { return gamma_; }
private:
    Model* draft_;
    Model* target_;
    float gamma_, min_gamma_, max_gamma_;
    Sampler sampler_;
    SamplerConfig sampler_cfg_;
    float acceptance_rate_ = 0.0f;
    float acc_ema_ = 0.6f; // EWMA of acceptance rate for adaptive gamma
    int accepted_count_ = 0;
    int total_count_ = 0;
    int adapt_interval_ = 10; // re-evaluate gamma every N calls
    int calls_since_adapt_ = 0;
    bool verify_tokens(const std::vector<int>& draft_tokens,
                       const Tensor& target_logits, int vocab_size);
    void adapt_gamma();
};

// D3: Continuous batching — dynamic request scheduling
struct BatchRequest { std::vector<int> tokens; int max_tokens; int id; };
struct BatchResponse { std::string text; int id; };
class ContinuousBatching {
public:
    ContinuousBatching(Model* model, int max_batch = 8);
    void add_request(const BatchRequest& req);
    BatchResponse step();
    bool has_pending() const;
private:
    Model* model_;
    int max_batch_;
    std::queue<BatchRequest> queue_;
    std::vector<BatchRequest> active_;
    std::vector<std::vector<int>> outputs_;
    std::vector<KVCache> kv_caches_;
    Tensor build_attention_mask(int64_t B, int64_t S,
                                const std::vector<int>& seq_lens) const;
};

// D4: KV cache compression — OIL4/ternary for K/V storage
class CompressedKVCache {
public:
    CompressedKVCache(int64_t max_seq, int64_t n_layers, int64_t head_dim);
    void append(int layer, const Tensor& k, const Tensor& v);
    Tensor get_k(int layer, int64_t pos) const;
    Tensor get_v(int layer, int64_t pos) const;
    int64_t size() const { return seq_len_; }
    void clear();
private:
    int64_t max_seq_, n_layers_, head_dim_;
    int64_t seq_len_ = 0;
    struct CompressedBlock { std::vector<uint8_t> k_data; std::vector<uint8_t> v_data; };
    std::vector<std::vector<CompressedBlock>> k_blocks_, v_blocks_;
};

// D5: Prefix caching — share KV cache across requests with common prefix
class PrefixCache {
public:
    PrefixCache(int64_t layer_count);
    int64_t match_prefix(const std::vector<int>& tokens);
    void store(const std::vector<int>& tokens, int64_t cache_id);
    KVCache* get_cache(int64_t id, int64_t layer);
private:
    int64_t layers_;
    struct PrefixEntry { std::vector<int> prefix; KVCache cache; };
    std::vector<PrefixEntry> entries_;
};

// D6: Token tree decoding — branch-and-verify parallel decoding
class TreeDecoder {
public:
    TreeDecoder(Model* model, int beam_width = 4);
    std::vector<int> decode(const std::vector<int>& prompt, int max_tokens);
private:
    Model* model_;
    int beam_width_;
    struct Node { int token; float score; Node* parent; std::vector<Node*> children; };
    void expand_node(Node* n, int depth, int max_depth);
    void prune_tree(std::vector<Node*>& candidates);
    void delete_subtree(Node* n);
};

// D7: Flash decoding — parallel KV reduction for multi-turn
Tensor flash_decoding(const Tensor& Q, const Tensor& K, const Tensor& V, int64_t block_size = 64);

// D8: INT8 inference — quantized forward pass
class INT8Inference {
public:
    INT8Inference(Model* model);
    Tensor forward(const Tensor& input, const Tensor& positions);
private:
    Model* model_;
    bool is_quantized_ = false;
};

// D9: FP8 inference — FP8 GEMM forward pass with two-stage residual accumulation (FA-3 style)
class FP8Inference {
public:
    FP8Inference(Model* model);
    Tensor forward(const Tensor& input, const Tensor& positions);
    // Two-stage residual accumulation: dequant + residual tracking for long context
    Tensor forward_two_stage(const Tensor& input, const Tensor& positions);
private:
    Model* model_;
    static constexpr int64_t BLOCK = 64;
    void fp8_residual_accum(const float* src, const uint8_t* fp8_data,
                            const float* scales, int64_t num_blocks,
                            float* out, int64_t n);
};

// D10: Model sharding — split across devices and load only partial model
class ModelShard {
public:
    ModelShard(Model* model, int64_t start_layer, int64_t end_layer);
    Tensor forward(const Tensor& input);
private:
    Model* model_;
    int64_t start_, end_;
};

// D11: Dynamic batching — variable batch sizes
class DynamicBatcher {
public:
    DynamicBatcher(Model* model);
    std::vector<std::string> batch_generate(
        const std::vector<std::string>& prompts, int max_tokens);
private:
    Model* model_;
};

// D12: Request scheduling — priority queue with deadlines
struct Request { int id; std::vector<int> tokens; int priority; int64_t deadline; };
class RequestScheduler {
public:
    void add(const Request& req);
    Request next();
    bool has_next() const;
    int pending_count() const { return (int)queue_.size(); }
private:
    struct Compare { bool operator()(const Request& a, const Request& b); };
    std::priority_queue<Request, std::vector<Request>, Compare> queue_;
};

// D13: Memory management — pool allocator for inference
class InferenceMemoryPool {
public:
    InferenceMemoryPool(size_t block_size, int64_t num_blocks);
    void* alloc();
    void free(void* ptr);
    int64_t used() const { return used_; }
    int64_t capacity() const { return capacity_; }
private:
    size_t block_size_;
    int64_t capacity_, used_ = 0;
    std::vector<char> pool_;
    std::vector<bool> free_list_;
    std::mutex mtx_;
};

// D14-D16: Stream, stop, logprob
struct StreamConfig {
    bool stream = false;
    int max_tokens = 512;
    int eos_id = -1;
    bool stop_on_newline = false;
    std::vector<std::string> stop_sequences;
};

struct DecodeResult {
    std::string text;
    std::vector<int> tokens;
    std::vector<std::vector<float>> logprobs; // token × vocab
    float perplexity = 0;
};

// D17: Logprob output
std::vector<std::vector<float>> compute_logprobs(const Tensor& logits, const std::vector<int>& tokens);

// D18: Embedding endpoint
class EmbeddingEndpoint {
public:
    EmbeddingEndpoint(Model* model);
    Tensor embed(const std::string& text);
    Tensor embed_batch(const std::vector<std::string>& texts);
private:
    Model* model_;
};

// D19: Reranking — score query-document pairs
class Reranker {
public:
    Reranker(Model* model);
    float score(const std::string& query, const std::string& document);
    std::vector<float> score_batch(const std::string& query,
                                    const std::vector<std::string>& documents);
private:
    Model* model_;
};

// D20: Grammar decoding — enforce JSON/regex constraint
class GrammarDecoder {
public:
    GrammarDecoder(const std::string& grammar_file);
    std::vector<int> constrain(const std::vector<float>& logits,
                                const std::vector<int>& prefix);
private:
    std::vector<std::vector<bool>> allowed_tokens_;
    int vocab_size_ = 32000;
    bool matches_prefix(const std::vector<int>& prefix) const;
    void parse_grammar(const std::string& grammar_file);
};

} // namespace oil
