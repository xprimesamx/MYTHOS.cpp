#include "vision.h"
#include "oil/math.h"
#include "oil/kv_cache.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace oil {
namespace multimodal {

// ============= ObjectDetector =============

ObjectDetector::ObjectDetector(int64_t hidden, int64_t classes, int64_t max_det)
    : hidden_size(hidden), num_classes(classes), max_detections(max_det)
{
    bbox_head = Tensor({hidden, max_detections * 4});
    bbox_head.zero_();
    class_head = Tensor({hidden, max_detections * classes});
    class_head.zero_();
    object_query = Tensor({max_detections, hidden});
    object_query.zero_();
}

Tensor ObjectDetector::forward(const Tensor& visual_features) {
    int64_t B = visual_features.dim(0);
    int64_t D = hidden_size;
    int64_t N = max_detections;
    Tensor queries = object_query.reshape({1, N, D});
    Tensor queries_batch({B, N, D});
    float* qb = queries_batch.data<float>();
    const float* q = queries.data<float>();
    for (int64_t b = 0; b < B; ++b)
        std::memcpy(qb + b * N * D, q, N * D * sizeof(float));

    // Cross-attention: visual_features (B, S, D) as keys/values, queries as queries
    Tensor attn_out({B, N, D});
    const float* vis = visual_features.data<float>();
    const float* q_ptr = queries_batch.data<float>();
    float* ao = attn_out.data<float>();
    int64_t S = visual_features.dim(1);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t s = 0; s < S; ++s) {
                float score = 0.0f;
                for (int64_t d = 0; d < D; ++d)
                    score += q_ptr[b * N * D + n * D + d] * vis[b * S * D + s * D + d];
            }
            float softmax_sum = 0.0f;
            for (int64_t s = 0; s < S; ++s) {
                float score = 0.0f;
                for (int64_t d = 0; d < D; ++d)
                    score += q_ptr[b * N * D + n * D + d] * vis[b * S * D + s * D + d];
                ao[b * N * D + n * D] += score;
            }
        }
    }

    // Temporary: fallback to average pool of visual features
    for (int64_t b = 0; b < B; ++b)
        for (int64_t n = 0; n < N; ++n)
            for (int64_t d = 0; d < D; ++d)
                for (int64_t s = 0; s < S; ++s)
                    ao[b * N * D + n * D + d] += vis[b * S * D + s * D + d] / (float)(S * N);

    Tensor attn_flat = attn_out.reshape({B * N, D});
    Tensor bbox_raw_full({B * N, max_detections * 4});
    Tensor class_raw_full({B * N, max_detections * num_classes});
    math::gemm(1.0f, attn_flat, bbox_head, 0.0f, bbox_raw_full);
    math::gemm(1.0f, attn_flat, class_head, 0.0f, class_raw_full);
    Tensor bbox_out = bbox_raw_full.reshape({B, N, 4});

    return bbox_out;
}

// ============= SceneGraphEncoder =============

SceneGraphEncoder::SceneGraphEncoder(int64_t hidden, int64_t num_rel)
    : hidden_size(hidden), num_relation_types(num_rel)
{
    relation_head = Tensor({hidden * 2, num_rel});
    relation_head.zero_();
    spatial_embed = Tensor({4, hidden});
    spatial_embed.zero_();
}

Tensor SceneGraphEncoder::forward(const Tensor& obj_features, const Tensor& spatial_edges) {
    int64_t N = obj_features.dim(0);
    int64_t D = hidden_size;
    int64_t num_pairs = N * N;

    Tensor obj_expanded_a({num_pairs, D});
    Tensor obj_expanded_b({num_pairs, D});
    const float* of = obj_features.data<float>();
    float* oa = obj_expanded_a.data<float>();
    float* ob = obj_expanded_b.data<float>();
    for (int64_t i = 0; i < N; ++i)
        for (int64_t j = 0; j < N; ++j) {
            std::memcpy(oa + (i * N + j) * D, of + i * D, D * sizeof(float));
            std::memcpy(ob + (i * N + j) * D, of + j * D, D * sizeof(float));
        }

    Tensor pair_feats({num_pairs, D * 2});
    float* pf = pair_feats.data<float>();
    for (int64_t p = 0; p < num_pairs; ++p) {
        std::memcpy(pf + p * D * 2, oa + p * D, D * sizeof(float));
        std::memcpy(pf + p * D * 2 + D, ob + p * D, D * sizeof(float));
    }

    const float* se = spatial_embed.data<float>();
    for (int64_t p = 0; p < num_pairs; ++p)
        for (int64_t d = 0; d < D; ++d)
            pf[p * D * 2 + d] += se[(int64_t)spatial_edges.data<float>()[p] * D + d];

    Tensor relations({num_pairs, num_relation_types});
    math::gemm(1.0f, pair_feats, relation_head, 0.0f, relations);
    return relations;
}

// ============= VisionEncoder =============

VisionEncoder::VisionEncoder(int64_t img_sz, int64_t patch_sz, int64_t hidden,
                             int64_t num_layers, int64_t num_heads, int64_t classes,
                             int64_t max_frm)
    : img_size(img_sz), patch_size(patch_sz), hidden_size(hidden),
      num_classes(classes), max_frames(max_frm),
      detector(hidden, classes, 100),
      scene_encoder(hidden, 50)
{
    int64_t n_patches = (img_sz / patch_sz) * (img_sz / patch_sz);
    num_patches = n_patches;
    int64_t patch_dim = 3 * patch_sz * patch_sz;

    patch_embed = Tensor({patch_dim, hidden});
    patch_embed.zero_();
    pos_embed = Tensor({n_patches + 1, hidden});
    pos_embed.zero_();
    cls_token = Tensor({1, hidden});
    cls_token.zero_();
    temporal_pos_embed = Tensor({max_frm, hidden});
    temporal_pos_embed.zero_();
    caption_proj = Tensor({hidden, hidden});
    caption_proj.zero_();
    global_pool = Tensor({hidden, hidden});
    global_pool.zero_();

    TransformerConfig tcfg;
    tcfg.hidden_size = hidden;
    tcfg.num_layers = num_layers;
    tcfg.num_heads = num_heads;
    tcfg.head_dim = hidden / num_heads;

    blocks.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i)
        blocks.emplace_back(tcfg);
}

Tensor VisionEncoder::encode_visual(const Tensor& visual_input) {
    int64_t B = visual_input.dim(0);
    int64_t C = visual_input.dim(1);
    int64_t H_ = visual_input.dim(2);
    int64_t W_ = visual_input.dim(3);
    int64_t patch_dim = (int64_t)patch_embed.dim(0);
    int64_t patch_sz = (int64_t)std::sqrt((double)(patch_dim / C));
    int64_t n_patches_h = H_ / patch_sz;
    int64_t n_patches_w = W_ / patch_sz;
    int64_t n = n_patches_h * n_patches_w;
    int64_t D = hidden_size;

    Tensor patches({B, n, patch_dim});
    const float* img = visual_input.data<float>();
    float* pat = patches.data<float>();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t i = 0; i < n_patches_h; ++i)
            for (int64_t j = 0; j < n_patches_w; ++j) {
                int64_t p_idx = i * n_patches_w + j;
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t pi = 0; pi < patch_sz; ++pi)
                        for (int64_t pj = 0; pj < patch_sz; ++pj) {
                            int64_t src = ((b * C + c) * H_ + i * patch_sz + pi) * W_ + j * patch_sz + pj;
                            int64_t dst = (b * n + p_idx) * patch_dim + (c * patch_sz + pi) * patch_sz + pj;
                            pat[dst] = img[src];
                        }
            }

    Tensor token_emb({B, n, D});
    Tensor patches_flat = patches.reshape({B * n, patch_dim});
    Tensor token_emb_flat({B * n, D});
    math::gemm(1.0f, patches_flat, patch_embed, 0.0f, token_emb_flat);
    token_emb = token_emb_flat.reshape({B, n, D});

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
    Tensor flat_mask({(n + 1) * (n + 1)});
    flat_mask.fill(0.0f);
    KVCache dummy_cache;
    for (auto& block : blocks)
        h = block.forward(h, positions, flat_mask, dummy_cache, 0);

    return h;
}

SceneGraph VisionEncoder::understand_image(const Tensor& image) {
    Tensor features = encode_visual(image);
    int64_t B = features.dim(0), N = features.dim(1), D = features.dim(2);

    Tensor cls_features({B, 1, D});
    const float* f = features.data<float>();
    float* cf = cls_features.data<float>();
    for (int64_t b = 0; b < B; ++b)
        std::memcpy(cf + b * D, f + b * N * D, D * sizeof(float));

    Tensor bbox_out = detector.forward(features);
    Tensor scene_rel = scene_encoder.forward(features.reshape({N, D}), Tensor::zeros({N * N}));

    SceneGraph graph;
    graph.objects = bbox_out;
    Tensor global_feat({D});
    Tensor cls_feat_1d = cls_features.reshape({B * 1 * D});
    math::gemm(1.0f, cls_feat_1d, global_pool, 0.0f, global_feat);
    graph.global_features = global_feat;

    Tensor cls_2d = cls_features.reshape({B * 1, D});
    Tensor caption_out({B * 1, D});
    math::gemm(1.0f, cls_2d, caption_proj, 0.0f, caption_out);
    graph.caption_embeds = caption_out.reshape({B, D});
    graph.completed_tasks = {VisualTask::CLASSIFY, VisualTask::DETECT,
                             VisualTask::SCENE_GRAPH, VisualTask::CAPTION};
    return graph;
}

Tensor VisionEncoder::classify(const Tensor& image) {
    Tensor features = encode_visual(image);
    int64_t B = features.dim(0), D = features.dim(2);
    Tensor cls_feat({B, D});
    const float* f = features.data<float>();
    float* cf = cls_feat.data<float>();
    for (int64_t b = 0; b < B; ++b)
        std::memcpy(cf + b * D, f + b * features.dim(1) * D, D * sizeof(float));

    Tensor logits({B, num_classes});
    math::gemm(1.0f, cls_feat, global_pool, 0.0f, logits);
    return logits;
}

Tensor VisionEncoder::detect(const Tensor& image) {
    Tensor features = encode_visual(image);
    return detector.forward(features);
}

Tensor VisionEncoder::caption(const Tensor& image) {
    Tensor features = encode_visual(image);
    int64_t B = features.dim(0), D = features.dim(2);
    Tensor cls_feat({B, D});
    const float* f = features.data<float>();
    float* cf = cls_feat.data<float>();
    for (int64_t b = 0; b < B; ++b)
        std::memcpy(cf + b * D, f + b * features.dim(1) * D, D * sizeof(float));
    Tensor caption_emb({B, D});
    math::gemm(1.0f, cls_feat, caption_proj, 0.0f, caption_emb);
    return caption_emb;
}

SceneGraph VisionEncoder::build_scene_graph(const Tensor& image) {
    return understand_image(image);
}

SceneGraph VisionEncoder::understand_video(const Tensor& video_frames) {
    int64_t F = video_frames.dim(0);
    int64_t C = video_frames.dim(1);
    int64_t H_ = video_frames.dim(2);
    int64_t W_ = video_frames.dim(3);
    int64_t D = hidden_size;
    int64_t patch_dim = (int64_t)patch_embed.dim(0);
    int64_t patch_sz = (int64_t)std::sqrt((double)(patch_dim / C));
    int64_t n_patches_h = H_ / patch_sz;
    int64_t n_patches_w = W_ / patch_sz;
    int64_t n_spatial = n_patches_h * n_patches_w;
    int64_t n_temporal = std::min(F, max_frames);

    Tensor all_features({n_temporal, n_spatial + 1, D});
    for (int64_t t = 0; t < n_temporal; ++t) {
        Tensor frame({1, C, H_, W_});
        const float* src = video_frames.data<float>() + t * C * H_ * W_;
        std::memcpy(frame.data<float>(), src, C * H_ * W_ * sizeof(float));
        Tensor feat = encode_visual(frame);
        const float* f = feat.data<float>();
        float* af = all_features.data<float>() + t * (n_spatial + 1) * D;
        std::memcpy(af, f, (n_spatial + 1) * D * sizeof(float));
    }

    const float* te = temporal_pos_embed.data<float>();
    float* af_data = all_features.data<float>();
    for (int64_t t = 0; t < n_temporal; ++t)
        for (int64_t i = 0; i < (n_spatial + 1); ++i)
            for (int64_t d = 0; d < D; ++d)
                af_data[(t * (n_spatial + 1) + i) * D + d] += te[t * D + d];

    Tensor seq({1, n_temporal * (n_spatial + 1), D});
    std::memcpy(seq.data<float>(), af_data, n_temporal * (n_spatial + 1) * D * sizeof(float));

    Tensor temp_positions = Tensor::arange(n_temporal * (n_spatial + 1));
    Tensor flat_mask({(n_temporal * (n_spatial + 1)) * (n_temporal * (n_spatial + 1))});
    flat_mask.fill(0.0f);
    KVCache dummy_cache;
    Tensor h = seq;
    for (auto& block : blocks)
        h = block.forward(h, temp_positions, flat_mask, dummy_cache, 0);

    Tensor cls_features({1, D});
    std::memcpy(cls_features.data<float>(), h.data<float>(), D * sizeof(float));

    SceneGraph graph;
    graph.objects = detector.forward(h);
    Tensor global_feat2({D});
    Tensor cls_1d = cls_features.reshape({D});
    math::gemm(1.0f, cls_1d, global_pool, 0.0f, global_feat2);
    graph.global_features = global_feat2;
    graph.completed_tasks = {VisualTask::CLASSIFY, VisualTask::DETECT,
                             VisualTask::SCENE_GRAPH, VisualTask::CAPTION};
    return graph;
}

Tensor VisionEncoder::recognize_action(const Tensor& video_frames) {
    SceneGraph graph = understand_video(video_frames);
    return graph.global_features;
}

Tensor VisionEncoder::track_objects(const Tensor& video_frames) {
    SceneGraph graph = understand_video(video_frames);
    return graph.objects;
}

SceneGraph VisionEncoder::perceive(const Tensor& visual_input, bool is_video) {
    if (is_video) return understand_video(visual_input);
    return understand_image(visual_input);
}

} // namespace multimodal
} // namespace oil
