# `generator.h` — Text Generation Pipeline

**Path:** `include/oil/generator.h`

High-level text generation pipeline combining model, tokenizer, sampler, and KV cache.

## Generator Class

```cpp
class Generator {
    std::unique_ptr<Model> model;
    std::unique_ptr<Tokenizer> tokenizer;
    std::unique_ptr<Sampler> sampler;
    std::unique_ptr<KVCache> cache;

    Generator(std::unique_ptr<Model> model, 
              std::unique_ptr<Tokenizer> tokenizer);
    
    std::string generate(const std::string& prompt, int max_tokens = 256);
    std::vector<int64_t> generate_ids(const std::vector<int64_t>& input_ids, 
                                      int max_tokens = 256);
    void stream_generate(const std::string& prompt, 
                         std::function<void(const std::string&)> callback);
};
```

### Methods

| Method | Description |
|--------|-------------|
| `generate(prompt, max_tokens)` | Generate text from string prompt |
| `generate_ids(input_ids, max_tokens)` | Generate from token IDs |
| `stream_generate(prompt, callback)` | Stream tokens one by one |

### Generation Process

1. Tokenize prompt → token IDs
2. Encode positions
3. For each step:
   a. Run model forward pass
   b. Sample next token from logits
   c. If EOS token → stop
   d. Append token to sequence
4. Decode token IDs → output string

### Configuration

The generator respects model config, sampler settings, and supports streaming output via callback.
