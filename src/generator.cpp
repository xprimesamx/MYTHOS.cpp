#include "oil/generator.h"
#include "oil/tokenizer.h"
#include "oil/model.h"
#include "oil/sampler.h"
#include <chrono>
#include <sstream>

namespace oil {

Generator::Generator(Model* model, Tokenizer* tokenizer)
    : model_(model), tokenizer_(tokenizer), sampler_(42) {}

std::vector<int> Generator::generate_tokens(const std::vector<int>& input_ids,
                                              const SamplerConfig& cfg) {
    int64_t seq_len = input_ids.size();
    int64_t B = 1;
    
    Tensor input_tensor(Shape{B, seq_len}, DType::F32);
    int* id_ptr = (int*)input_tensor.data();
    for (size_t i = 0; i < input_ids.size(); i++) id_ptr[i] = input_ids[i];
    
    Tensor positions(Shape{B, seq_len}, DType::F32);
    int* pos_ptr = (int*)positions.data();
    for (int64_t i = 0; i < seq_len; i++) pos_ptr[i] = (int)i;
    
    // Prefill
    Tensor logits = model_->forward(input_tensor, positions);
    int64_t vocab = logits.shape().dims[2];
    float* last_logits = (float*)logits.data() + (seq_len - 1) * vocab;
    int next_token = sampler_.sample(last_logits, (int)vocab, cfg);
    
    std::vector<int> output_ids(input_ids);
    output_ids.push_back(next_token);
    
    // Decode loop
    for (int t = 0; t < cfg.max_tokens - 1; t++) {
        Tensor single_input(Shape{B, 1}, DType::F32);
        ((int*)single_input.data())[0] = next_token;
        
        Tensor single_pos(Shape{B, 1}, DType::F32);
        ((int*)single_pos.data())[0] = (int)seq_len + t;
        
        logits = model_->forward(single_input, single_pos);
        float* out = (float*)logits.data();
        next_token = sampler_.sample(out, (int)vocab, cfg);
        output_ids.push_back(next_token);
        
        if (next_token == tokenizer_->eos_id()) break;
    }
    
    return output_ids;
}

std::string Generator::generate(const std::string& prompt, const SamplerConfig& cfg) {
    auto ids = tokenizer_->encode(prompt);
    auto output_ids = generate_tokens(ids, cfg);
    return tokenizer_->decode(output_ids);
}

void Generator::generate_stream(const std::string& prompt, const SamplerConfig& cfg,
                                 std::function<void(const std::string&)> on_token) {
    auto full_text = generate(prompt, cfg);
    on_token(full_text);
}

GenerationResult Generator::generate_full(const std::string& prompt,
                                            const SamplerConfig& cfg) {
    auto start = std::chrono::high_resolution_clock::now();
    auto ids = tokenizer_->encode(prompt);
    auto output_ids = generate_tokens(ids, cfg);
    auto end = std::chrono::high_resolution_clock::now();
    
    GenerationResult result;
    result.token_ids = output_ids;
    result.text = tokenizer_->decode(output_ids);
    result.duration_sec = std::chrono::duration<float>(end - start).count();
    int new_tokens = (int)output_ids.size() - (int)ids.size();
    result.tokens_per_sec = (int)(new_tokens / result.duration_sec);
    result.truncated = new_tokens >= cfg.max_tokens;
    return result;
}

} // namespace oil
