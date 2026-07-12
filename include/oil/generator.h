#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/model.h"
#include "oil/sampler.h"
#include "oil/tokenizer.h"
#include "oil/kv_cache.h"
#include <vector>
#include <string>
#include <functional>

namespace oil {

struct GenerationResult {
    std::vector<int> token_ids;
    std::string text;
    float duration_sec;
    int tokens_per_sec;
    bool truncated;
};

class Generator {
public:
    Generator(Model* model, Tokenizer* tokenizer);
    
    // Generate tokens from a prompt
    std::vector<int> generate_tokens(const std::vector<int>& input_ids,
                                      const SamplerConfig& cfg);
    
    // Generate text from a string prompt
    std::string generate(const std::string& prompt, const SamplerConfig& cfg);
    
    // Generate with token-by-token callback
    void generate_stream(const std::string& prompt, const SamplerConfig& cfg,
                          std::function<void(const std::string&)> on_token);
    
    // Full result with stats
    GenerationResult generate_full(const std::string& prompt, const SamplerConfig& cfg);
    
private:
    Model* model_;
    Tokenizer* tokenizer_;
    Sampler sampler_;
    KVCache kv_cache_;
};

} // namespace oil
