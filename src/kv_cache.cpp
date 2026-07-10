#include "oil/kv_cache.h"
#include <cstring>

namespace oil {

KVCache::KVCache(int num_layers, int64_t max_seq_len, int64_t num_heads,
                  int64_t head_dim, bool quantized)
    : num_layers_(num_layers), max_seq_len_(max_seq_len),
      num_heads_(num_heads), head_dim_(head_dim), quantized_(quantized)
{
    init(num_layers, max_seq_len, num_heads, head_dim, quantized);
}

void KVCache::init(int num_layers, int64_t max_seq_len, int64_t num_heads,
                    int64_t head_dim, bool quantized) {
    num_layers_ = num_layers;
    max_seq_len_ = max_seq_len;
    num_heads_ = num_heads;
    head_dim_ = head_dim;
    quantized_ = quantized;
    caches_.resize(num_layers);
    for (int i = 0; i < num_layers; i++) {
        caches_[i].k = Tensor::zeros(Shape{1, num_heads_, max_seq_len_, head_dim_});
        caches_[i].v = Tensor::zeros(Shape{1, num_heads_, max_seq_len_, head_dim_});
        caches_[i].current_pos = 0;
    }
}

void KVCache::append(int layer, const Tensor& k, const Tensor& v) {
    if (layer >= num_layers_) return;
    auto& c = caches_[layer];
    int64_t seq_len = k.shape().dims[2];
    int64_t pos = c.current_pos;
    
    float* kdst = (float*)c.k.data();
    float* vdst = (float*)c.v.data();
    const float* ksrc = (const float*)k.data();
    const float* vsrc = (const float*)v.data();
    
    int64_t h = num_heads_;
    int64_t d = head_dim_;
    
    for (int64_t s = 0; s < seq_len && pos + s < max_seq_len_; s++) {
        for (int64_t hh = 0; hh < h; hh++) {
            int64_t dst_offset = hh * max_seq_len_ * d + (pos + s) * d;
            int64_t src_offset = hh * seq_len * d + s * d;
            memcpy(kdst + dst_offset, ksrc + src_offset, d * sizeof(float));
            memcpy(vdst + dst_offset, vsrc + src_offset, d * sizeof(float));
        }
    }
    c.current_pos += seq_len;
}

std::pair<Tensor, Tensor> KVCache::get_range(int layer, int start, int end) const {
    if (layer >= num_layers_) return {};
    const auto& c = caches_[layer];
    int64_t len = (end - start > max_seq_len_) ? max_seq_len_ : end - start;
    Tensor k_out(Shape{1, num_heads_, len, head_dim_}, DType::F32);
    Tensor v_out(Shape{1, num_heads_, len, head_dim_}, DType::F32);
    
    const float* ksrc = (const float*)c.k.data();
    const float* vsrc = (const float*)c.v.data();
    float* kdst = (float*)k_out.data();
    float* vdst = (float*)v_out.data();
    
    for (int64_t s = 0; s < len; s++) {
        int64_t src_pos = start + s;
        if (src_pos >= max_seq_len_) break;
        for (int64_t h = 0; h < num_heads_; h++) {
            int64_t src_off = h * max_seq_len_ * head_dim_ + src_pos * head_dim_;
            int64_t dst_off = h * len * head_dim_ + s * head_dim_;
            memcpy(kdst + dst_off, ksrc + src_off, head_dim_ * sizeof(float));
            memcpy(vdst + dst_off, vsrc + src_off, head_dim_ * sizeof(float));
        }
    }
    return {k_out, v_out};
}

std::pair<Tensor, Tensor> KVCache::get_all(int layer) const {
    return get_range(layer, 0, caches_[layer].current_pos);
}

int KVCache::context_len() const {
    return caches_.empty() ? 0 : caches_[0].current_pos;
}

int KVCache::max_seq_len() const { return (int)max_seq_len_; }

size_t KVCache::size_bytes() const {
    size_t total = 0;
    for (auto& c : caches_) {
        total += c.k.size_bytes() + c.v.size_bytes();
    }
    return total;
}

void KVCache::clear() {
    for (auto& c : caches_) {
        c.k.zero_();
        c.v.zero_();
        c.current_pos = 0;
    }
}

void KVCache::resize(int64_t new_max_seq_len) {
    max_seq_len_ = new_max_seq_len;
    for (auto& c : caches_) {
        c.k = Tensor::zeros(Shape{1, num_heads_, max_seq_len_, head_dim_});
        c.v = Tensor::zeros(Shape{1, num_heads_, max_seq_len_, head_dim_});
        c.current_pos = 0;
    }
}

} // namespace oil
