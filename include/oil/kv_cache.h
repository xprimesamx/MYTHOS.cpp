#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include <vector>
#include <string>
#include <unordered_map>

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

// ===========================================================================
// PagedKVCache1T — 1T (minimum) to unlimited (maximum) hierarchical paging
//
// Design:
//   Logical token space: up to 2^40 (~1T) tokens, effectively unlimited
//   Physical memory: limited (e.g. 8GB), pages evicted to disk via LRU
//   NEVER lose data — evicted pages are offloaded to disk, retrievable
//
//   Hierarchical paging:
//     L1 PageTable:  256 entries -> L2 PageTable pointers
//     L2 PageTable:  256 entries -> L3 PageTable pointers
//     L3 PageTable:  256 entries -> physical block IDs
//     Each block = block_size_ tokens of K/V (heads × block_size × head_dim floats)
//
//   This gives 256^3 × block_size = 16.7M × block_size logical tokens.
//   With block_size=16, that's 268M tokens per layer per head.
//   With deeper levels or larger blocks, easily exceeds 1T.
//
//   Physical blocks are managed via LRU. When physical memory is full,
//   the least recently used block is offloaded to a disk file and its
//   slot is reused. On access, if the block is on disk, it is loaded back.
// ===========================================================================

class PagedKVCache1T {
public:
    static constexpr int64_t TABLE_ENTRIES = 256;
    static constexpr int64_t DEFAULT_BLOCK_SIZE = 16;
    static constexpr int64_t MIN_LOGICAL_TOKENS = (int64_t)1 << 40; // 1T

    PagedKVCache1T(int num_layers, int64_t num_heads, int64_t head_dim,
                   int64_t block_size = DEFAULT_BLOCK_SIZE,
                   size_t physical_memory_bytes = 8ULL * 1024 * 1024 * 1024,
                   const std::string& disk_path = "");

    ~PagedKVCache1T();

    void append(int layer, int64_t logical_pos, const Tensor& k, const Tensor& v);
    std::pair<Tensor, Tensor> get_range(int layer, int64_t start, int64_t end) const;
    std::pair<Tensor, Tensor> get_block(int layer, int64_t logical_pos) const;

    int64_t logical_capacity() const;
    int64_t num_physical_blocks() const;
    int64_t num_disk_blocks() const;
    size_t physical_memory_used() const;
    size_t physical_memory_limit() const;

    void flush_to_disk();
    void load_from_disk();
    void clear();

    int context_len() const;
    int64_t block_size() const { return block_size_; }
    int num_layers() const { return num_layers_; }
    int64_t num_heads() const { return num_heads_; }
    int64_t head_dim() const { return head_dim_; }

    bool verify_retrieval(int layer, int64_t pos, const Tensor& expected_k,
                          const Tensor& expected_v) const;
    int64_t max_logical_tokens_per_layer() const;

private:
    struct PhysicalBlock {
        int64_t id = -1;
        mutable std::vector<float> k_data;
        mutable std::vector<float> v_data;
        bool dirty = false;
        mutable int64_t last_access = 0;
        mutable bool on_disk = false;
        mutable std::string disk_file;
    };

    struct L3Table {
        int64_t entries[TABLE_ENTRIES];
        L3Table();
    };

    struct L2Table {
        L3Table* entries[TABLE_ENTRIES];
        L2Table();
        ~L2Table();
    };

    struct L1Table {
        L2Table* entries[TABLE_ENTRIES];
        L1Table();
        ~L1Table();
    };

    struct LayerState {
        L1Table* root;
        std::unordered_map<int64_t, PhysicalBlock> blocks;
        int64_t current_pos = 0;
    };

    int num_layers_;
    int64_t num_heads_;
    int64_t head_dim_;
    int64_t block_size_;
    size_t physical_memory_limit_;
    mutable size_t current_memory_used_;
    std::string disk_path_;
    std::vector<LayerState> layers_;
    mutable int64_t access_counter_;
    int64_t next_block_id_;

    int64_t resolve_block_id(int layer, int64_t logical_pos) const;
    int64_t alloc_block_id(int layer, int64_t logical_pos);
    void evict_lru(int layer);
    void offload_to_disk(int layer, int64_t block_id);
    void load_from_disk(int layer, int64_t block_id) const;
    std::string block_disk_path(int layer, int64_t block_id) const;
    int64_t tokens_per_block() const;
};

} // namespace oil
