#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include <vector>

namespace oil {

class KVCache {
public:
    KVCache() = default;
    KVCache(int num_layers, int64_t max_seq_len, int64_t num_heads, 
            int64_t head_dim, bool quantized = false);
    
    void init(int num_layers, int64_t max_seq_len, int64_t num_heads,
              int64_t head_dim, bool quantized = false);
    
    // Append a K,V pair for a layer at current position
    void append(int layer, const Tensor& k, const Tensor& v);
    
    // Get cached K,V for given position range
    std::pair<Tensor, Tensor> get_range(int layer, int start, int end) const;
    std::pair<Tensor, Tensor> get_all(int layer) const;
    
    // Get total context length
    int context_len() const;
    int max_seq_len() const;
    
    // Memory usage
    size_t size_bytes() const;
    
    // Clear and reset
    void clear();
    
    // Resize cache
    void resize(int64_t new_max_seq_len);
    
    static constexpr int FP8_BLOCK_SIZE = 64;
    static constexpr float FP8_MAX = 127.0f;

    static void quantize_fp8_block(const float* src, uint8_t* dst, float* scale,
                                    int64_t n);
    static void dequantize_fp8_block(const uint8_t* src, float scale,
                                      float* dst, int64_t n);

private:
    struct LayerCache {
        Tensor k;
        Tensor v;
        int current_pos = 0;
        // Quantized storage (used only when quantized_ == true)
        std::vector<uint8_t> k_quant;
        std::vector<uint8_t> v_quant;
        std::vector<float> k_scales;
        std::vector<float> v_scales;
    };
    std::vector<LayerCache> caches_;
    int num_layers_ = 0;
    int64_t max_seq_len_ = 0;
    int64_t num_heads_ = 0;
    int64_t head_dim_ = 0;
    bool quantized_ = false;
};

} // namespace oil
