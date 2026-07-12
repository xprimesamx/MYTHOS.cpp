#include "oil/kv_cache.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace oil {

void KVCache::quantize_fp8_block(const float* src, uint8_t* dst, float* scale,
                                  int64_t n) {
    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float a = std::fabs(src[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) {
        *scale = 1.0f;
        std::memset(dst, 0, (size_t)n);
        return;
    }
    *scale = max_abs / FP8_MAX;
    float inv_scale = 1.0f / *scale;
    for (int64_t i = 0; i < n; i++) {
        float q = src[i] * inv_scale;
        q = std::max(-FP8_MAX, std::min(FP8_MAX, q));
        dst[i] = (uint8_t)(int8_t)std::round(q);
    }
}

void KVCache::dequantize_fp8_block(const uint8_t* src, float scale,
                                     float* dst, int64_t n) {
    if (scale < 1e-10f) {
        std::memset(dst, 0, (size_t)n * sizeof(float));
        return;
    }
    for (int64_t i = 0; i < n; i++)
        dst[i] = (float)(int8_t)src[i] * scale;
}

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
        auto& c = caches_[i];
        int64_t h = num_heads_;
        int64_t d = head_dim_;
        c.k = Tensor::zeros(Shape{1, h, max_seq_len_, d});
        c.v = Tensor::zeros(Shape{1, h, max_seq_len_, d});
        c.current_pos = 0;
        if (quantized_) {
            int64_t total_vals = h * max_seq_len_ * d;
            int64_t num_blocks = (total_vals + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
            c.k_quant.resize((size_t)total_vals);
            c.v_quant.resize((size_t)total_vals);
            c.k_scales.resize((size_t)num_blocks);
            c.v_scales.resize((size_t)num_blocks);
        }
    }
}

void KVCache::append(int layer, const Tensor& k, const Tensor& v) {
    if (layer >= num_layers_) return;
    auto& c = caches_[layer];
    int64_t seq_len = k.shape().dims[2];
    int64_t pos = c.current_pos;
    int64_t h = num_heads_;
    int64_t d = head_dim_;

    if (quantized_) {
        int64_t total_vals = h * seq_len * d;
        int64_t num_blocks = (total_vals + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
        std::vector<uint8_t> k_tmp((size_t)total_vals);
        std::vector<uint8_t> v_tmp((size_t)total_vals);
        std::vector<float> k_scales_tmp((size_t)num_blocks);
        std::vector<float> v_scales_tmp((size_t)num_blocks);

        const float* ksrc = k.data<float>();
        const float* vsrc = v.data<float>();

        for (int64_t b = 0; b < num_blocks; b++) {
            int64_t start = b * FP8_BLOCK_SIZE;
            int64_t end = std::min(start + FP8_BLOCK_SIZE, total_vals);
            int64_t n = end - start;
            quantize_fp8_block(ksrc + start, k_tmp.data() + start,
                                &k_scales_tmp[(size_t)b], n);
            quantize_fp8_block(vsrc + start, v_tmp.data() + start,
                                &v_scales_tmp[(size_t)b], n);
        }

        int64_t dst_off = (int64_t)c.current_pos * d * h;
        int64_t copy_n = total_vals;
        if (dst_off + copy_n > (int64_t)c.k_quant.size())
            copy_n = (int64_t)c.k_quant.size() - dst_off;
        if (copy_n > 0) {
            std::memcpy(c.k_quant.data() + dst_off, k_tmp.data(), (size_t)copy_n);
            std::memcpy(c.v_quant.data() + dst_off, v_tmp.data(), (size_t)copy_n);
            int64_t scale_off = dst_off / FP8_BLOCK_SIZE;
            int64_t num_scales = (copy_n + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
            std::memcpy(c.k_scales.data() + scale_off, k_scales_tmp.data(),
                        (size_t)num_scales * sizeof(float));
            std::memcpy(c.v_scales.data() + scale_off, v_scales_tmp.data(),
                        (size_t)num_scales * sizeof(float));
        }
    } else {
        float* kdst = (float*)c.k.data();
        float* vdst = (float*)c.v.data();
        const float* ksrc = (const float*)k.data();
        const float* vsrc = (const float*)v.data();

        for (int64_t s = 0; s < seq_len && pos + s < max_seq_len_; s++) {
            for (int64_t hh = 0; hh < h; hh++) {
                int64_t dst_offset = hh * max_seq_len_ * d + (pos + s) * d;
                int64_t src_offset = hh * seq_len * d + s * d;
                memcpy(kdst + dst_offset, ksrc + src_offset, (size_t)d * sizeof(float));
                memcpy(vdst + dst_offset, vsrc + src_offset, (size_t)d * sizeof(float));
            }
        }
    }
    c.current_pos += (int)seq_len;
}

std::pair<Tensor, Tensor> KVCache::get_range(int layer, int start, int end) const {
    if (layer >= num_layers_) return {};
    const auto& c = caches_[layer];
    int64_t len = (end - start > max_seq_len_) ? max_seq_len_ : end - start;
    Tensor k_out(Shape{1, num_heads_, len, head_dim_});
    Tensor v_out(Shape{1, num_heads_, len, head_dim_});

    if (quantized_) {
        int64_t h = num_heads_;
        int64_t d = head_dim_;
        for (int64_t s = 0; s < len; s++) {
            int64_t src_pos = start + s;
            if (src_pos >= max_seq_len_) break;
            for (int64_t hh = 0; hh < h; hh++) {
                int64_t read_off = hh * max_seq_len_ * d + src_pos * d;
                int64_t write_off = hh * len * d + s * d;
                float* kdst = k_out.data<float>();
                float* vdst = v_out.data<float>();
                for (int64_t i = 0; i < d; i++) {
                    int64_t off = read_off + i;
                    int64_t bi = off / FP8_BLOCK_SIZE;
                    int64_t bo = off % FP8_BLOCK_SIZE;
                    kdst[write_off + i] = (float)(int8_t)c.k_quant[(size_t)off] * c.k_scales[(size_t)bi];
                    vdst[write_off + i] = (float)(int8_t)c.v_quant[(size_t)off] * c.v_scales[(size_t)bi];
                }
            }
        }
    } else {
        const float* ksrc = (const float*)c.k.data();
        const float* vsrc = (const float*)c.v.data();
        float* kdst = (float*)k_out.data();
        float* vdst = (float*)v_out.data();

        for (int64_t s = 0; s < len; s++) {
            int64_t src_pos = start + s;
            if (src_pos >= max_seq_len_) break;
            for (int64_t hh = 0; hh < num_heads_; hh++) {
                int64_t src_off = hh * max_seq_len_ * head_dim_ + src_pos * head_dim_;
                int64_t dst_off = hh * len * head_dim_ + s * head_dim_;
                memcpy(kdst + dst_off, ksrc + src_off, (size_t)head_dim_ * sizeof(float));
                memcpy(vdst + dst_off, vsrc + src_off, (size_t)head_dim_ * sizeof(float));
            }
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
        if (quantized_) {
            total += c.k_quant.size() + c.v_quant.size();
            total += c.k_scales.size() * sizeof(float);
            total += c.v_scales.size() * sizeof(float);
        } else {
            total += c.k.size_bytes() + c.v.size_bytes();
        }
    }
    return total;
}

void KVCache::clear() {
    for (auto& c : caches_) {
        if (quantized_) {
            std::memset(c.k_quant.data(), 0, c.k_quant.size());
            std::memset(c.v_quant.data(), 0, c.v_quant.size());
            std::memset(c.k_scales.data(), 0, c.k_scales.size() * sizeof(float));
            std::memset(c.v_scales.data(), 0, c.v_scales.size() * sizeof(float));
        } else {
            c.k.zero_();
            c.v.zero_();
        }
        c.current_pos = 0;
    }
}

void KVCache::resize(int64_t new_max_seq_len) {
    max_seq_len_ = new_max_seq_len;
    for (auto& c : caches_) {
        c.k = Tensor::zeros(Shape{1, num_heads_, max_seq_len_, head_dim_});
        c.v = Tensor::zeros(Shape{1, num_heads_, max_seq_len_, head_dim_});
        c.current_pos = 0;
        int64_t total_vals = num_heads_ * max_seq_len_ * head_dim_;
        c.k_quant.resize((size_t)total_vals);
        c.v_quant.resize((size_t)total_vals);
        int64_t num_blocks = (total_vals + FP8_BLOCK_SIZE - 1) / FP8_BLOCK_SIZE;
        c.k_scales.resize((size_t)num_blocks);
        c.v_scales.resize((size_t)num_blocks);
    }
}

} // namespace oil
