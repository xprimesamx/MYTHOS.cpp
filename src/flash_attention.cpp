#include "oil/flash_attention.h"
#include "oil/math.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace oil {

FlashAttention::FlashAttention(const FlashAttentionConfig& cfg) : cfg_(cfg) {}

Tensor flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                               const Tensor& mask, float dropout_p) {
    int64_t B = Q.dim(0), H = Q.dim(1), N = Q.dim(2), D = Q.dim(3);
    float scale = 1.0f / std::sqrt((float)D);
    bool causal = true;

    Tensor output({B, H, N, D});
    output.zero_();
    float* out = output.data<float>();
    const float* q = Q.data<float>();
    const float* k = K.data<float>();
    const float* v = V.data<float>();
    const float* m = mask.data<float>();

    int64_t block = 64;
    std::vector<float> row_max(B * H * N, -INFINITY);
    std::vector<float> row_sum(B * H * N, 0.0f);

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t bh_off = (b * H + h) * N * D;
            const float* q_ptr = q + (b * H + h) * N * D;
            const float* k_ptr = k + (b * H + h) * N * D;
            const float* v_ptr = v + (b * H + h) * N * D;

            for (int64_t jb = 0; jb < N; jb += block) {
                int64_t j_end = std::min(jb + block, N);
                int64_t k_block = j_end - jb;

                std::vector<float> k_block_data(k_block * D);
                std::vector<float> v_block_data(k_block * D);
                for (int64_t j = 0; j < k_block; ++j) {
                    std::memcpy(&k_block_data[j * D], k_ptr + (jb + j) * D, D * sizeof(float));
                    std::memcpy(&v_block_data[j * D], v_ptr + (jb + j) * D, D * sizeof(float));
                }

                for (int64_t i = 0; i < N; ++i) {
                    if (causal && i < jb) continue;

                    int64_t idx = b * H * N + h * N + i;
                    float rm = row_max[idx];
                    float rs = row_sum[idx];
                    float* o_row = out + bh_off + i * D;

                    float qk_max = -INFINITY;
                    std::vector<float> scores(k_block);
                    for (int64_t j = 0; j < k_block; ++j) {
                        float dot = 0;
                        for (int64_t d = 0; d < D; ++d)
                            dot += q_ptr[i * D + d] * k_block_data[j * D + d];
                        float masked = dot * scale;
                        if (m) masked += m[i * N + (jb + j)];
                        if (causal && (jb + j) > i) masked = -INFINITY;
                        scores[j] = masked;
                        if (masked > qk_max) qk_max = masked;
                    }

                    float new_rm = std::max(rm, qk_max);
                    float exp_diff = rm != -INFINITY ? std::exp(rm - new_rm) : 0.0f;
                    rs *= exp_diff;

                    float sum_exp = 0;
                    for (int64_t j = 0; j < k_block; ++j) {
                        float e = std::exp(scores[j] - new_rm);
                        sum_exp += e;
                        for (int64_t d = 0; d < D; ++d)
                            o_row[d] *= exp_diff;
                        for (int64_t d = 0; d < D; ++d)
                            o_row[d] += e * v_block_data[j * D + d];
                    }
                    rs += sum_exp;
                    row_max[idx] = new_rm;
                    row_sum[idx] = rs;
                }
            }

            for (int64_t i = 0; i < N; ++i) {
                float inv_sum = 1.0f / (row_sum[b * H * N + h * N + i] + 1e-10f);
                float* o_row = out + bh_off + i * D;
                for (int64_t d = 0; d < D; ++d)
                    o_row[d] *= inv_sum;
            }
        }
    }
    return output;
}

Tensor FlashAttention::forward(const Tensor& Q, const Tensor& K,
                                const Tensor& V, const Tensor& mask) {
    return flash_attention_forward(Q, K, V, mask, 0.0f);
}

} // namespace oil
