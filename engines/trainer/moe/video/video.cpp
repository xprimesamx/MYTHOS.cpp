#include "video.h"
#include "oil/math.h"
#include "oil/kv_cache.h"

namespace oil {
namespace multimodal {

VideoEncoder::VideoEncoder(int64_t tube_size, int64_t img_size, int64_t patch_size,
                           int64_t hidden, int64_t num_layers, int64_t num_heads)
    : hidden_size(hidden)
{
    int64_t n_patches = (img_size / patch_size) * (img_size / patch_size);
    num_tubes = n_patches;
    int64_t tube_dim = 3 * tube_size * patch_size * patch_size;

    conv3d_proj = Tensor({tube_dim, hidden});
    conv3d_proj.zero_();
    pos_embed = Tensor({n_patches + 1, hidden});
    pos_embed.zero_();
    cls_token = Tensor({1, hidden});
    cls_token.zero_();

    TransformerConfig tcfg;
    tcfg.hidden_size = hidden;
    tcfg.num_layers = num_layers;
    tcfg.num_heads = num_heads;
    tcfg.head_dim = hidden / num_heads;

    blocks.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i)
        blocks.emplace_back(tcfg);
}

Tensor VideoEncoder::forward(const Tensor& video) {
    int64_t B = video.dim(0);
    int64_t C = video.dim(1);
    int64_t T_ = video.dim(2);
    int64_t H_ = video.dim(3);
    int64_t W_ = video.dim(4);
    int64_t D = hidden_size;
    int64_t n = num_tubes;

    Tensor tube_flat({B, n, conv3d_proj.dim(0)});
    tube_flat.zero_();

    Tensor token_emb({B, n, D});
    Tensor token_emb_2d = token_emb.reshape({B * n, D});
    math::gemm(1.0f, tube_flat.reshape({B * n, conv3d_proj.dim(0)}), conv3d_proj, 0.0f, token_emb_2d);

    Tensor seq({B, n + 1, D});
    const float* pe = pos_embed.data<float>();
    const float* cls = cls_token.data<float>();
    float* s = seq.data<float>();
    float* te = token_emb.data<float>();
    for (int64_t b = 0; b < B; ++b) {
        std::memcpy(s + (b * (n + 1)) * D, cls, D * sizeof(float));
        for (int64_t i = 0; i < D; ++i)
            s[(b * (n + 1)) * D + i] += pe[i];
        for (int64_t p = 0; p < n; ++p) {
            std::memcpy(s + (b * (n + 1) + p + 1) * D, te + (b * n + p) * D, D * sizeof(float));
            for (int64_t i = 0; i < D; ++i)
                s[(b * (n + 1) + p + 1) * D + i] += pe[(p + 1) * D + i];
        }
    }

    Tensor h = seq;
    Tensor positions = Tensor::arange(n + 1);
    Tensor mask({(n + 1) * (n + 1)});
    mask.fill(0.0f);
    KVCache dummy_cache;
    for (auto& block : blocks)
        h = block.forward(h, positions, mask, dummy_cache, 0);

    return h;
}

} // namespace multimodal
} // namespace oil
