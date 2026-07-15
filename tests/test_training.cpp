#include "oil/model.h"
#include "oil/autograd.h"
#include "oil/math.h"
#include "oil/random.h"
#include "oil/optimizer.h"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace oil;

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); g_failures++; } \
    else { printf("  PASS: %s\n", msg); } \
} while(0)

static void collect_all_params(DenseModel& dm, std::vector<Tensor*>& params) {
    params.push_back(&dm.tok_embeddings->weight);
    for (auto& layer : dm.layers) {
        params.push_back(&layer->attention_norm.weight);
        params.push_back(&layer->attention.q_proj.weight);
        if (layer->attention.q_proj.bias.numel() > 0)
            params.push_back(&layer->attention.q_proj.bias);
        params.push_back(&layer->attention.k_proj.weight);
        if (layer->attention.k_proj.bias.numel() > 0)
            params.push_back(&layer->attention.k_proj.bias);
        params.push_back(&layer->attention.v_proj.weight);
        if (layer->attention.v_proj.bias.numel() > 0)
            params.push_back(&layer->attention.v_proj.bias);
        params.push_back(&layer->attention.o_proj.weight);
        if (layer->attention.o_proj.bias.numel() > 0)
            params.push_back(&layer->attention.o_proj.bias);
        params.push_back(&layer->ffn_norm.weight);
        params.push_back(&layer->ffn.gate_proj.weight);
        if (layer->ffn.gate_proj.bias.numel() > 0)
            params.push_back(&layer->ffn.gate_proj.bias);
        params.push_back(&layer->ffn.up_proj.weight);
        if (layer->ffn.up_proj.bias.numel() > 0)
            params.push_back(&layer->ffn.up_proj.bias);
        params.push_back(&layer->ffn.down_proj.weight);
        if (layer->ffn.down_proj.bias.numel() > 0)
            params.push_back(&layer->ffn.down_proj.bias);
    }
    params.push_back(&dm.norm->weight);
    params.push_back(&dm.lm_head->weight);
    if (dm.lm_head->bias.numel() > 0)
        params.push_back(&dm.lm_head->bias);
}

static float compute_loss(DenseModel& m, const Tensor& inp, const Tensor& pos,
                          const Tensor& tgt) {
    Tensor logits = m.forward(inp, pos);
    int64_t Seq = logits.dim(1);
    int64_t Vocab = logits.dim(2);
    const float* ld = logits.data<float>();
    const float* tgt_d = tgt.data<float>();
    float loss = 0;
    for (int64_t s = 0; s < Seq; s++) {
        float max_l = -1e30f;
        for (int64_t v = 0; v < Vocab; v++)
            if (ld[s * Vocab + v] > max_l) max_l = ld[s * Vocab + v];
        float sum = 0;
        for (int64_t v = 0; v < Vocab; v++)
            sum += std::exp(ld[s * Vocab + v] - max_l);
        int64_t t = (int64_t)tgt_d[s];
        if (t >= 0 && t < Vocab)
            loss += -(ld[s * Vocab + t] - max_l - std::log(sum));
    }
    return loss / (float)Seq;
}

int main() {
    printf("=== Training Test ===\n\n");

    TransformerConfig cfg;
    cfg.hidden_size = 64;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.head_dim = 16;
    cfg.ffn_hidden_size = 128;
    cfg.vocab_size = 100;
    cfg.max_seq_len = 32;

    DenseModel model(cfg);
    printf("Model created: %lld params\n", (long long)model.param_count());
    CHECK(model.param_count() > 0, "Model has parameters");

    int64_t S = 8;
    int64_t V = cfg.vocab_size;
    Tensor input_ids(Shape{1, S});
    Tensor positions(Shape{1, S});
    Tensor target_ids(Shape{1, S});
    float* id = input_ids.data<float>();
    float* pd = positions.data<float>();
    float* td = target_ids.data<float>();

    for (int64_t i = 0; i < S; i++) {
        id[i] = (float)(i % V);
        pd[i] = (float)i;
        td[i] = (float)((i + 1) % V);
    }

    float initial_loss = compute_loss(model, input_ids, positions, target_ids);
    printf("Initial loss: %.4f\n", initial_loss);
    CHECK(initial_loss > 0, "Initial loss is positive");
    CHECK(std::isfinite(initial_loss), "Initial loss is finite");

    float lr = 0.001f;
    int num_steps = 20;
    std::vector<float> losses;

    std::vector<Tensor*> params;
    collect_all_params(model, params);

    auto& engine = AutogradEngine::instance();
    for (auto* p : params)
        engine.register_parameter(p);

    SGD optimizer(lr);
    optimizer.add_param_group(params);

    for (int step = 0; step < num_steps; step++) {
        optimizer.zero_grad();
        AutogradEngine::set_enabled(true);

        Tensor logits = model.forward(input_ids, positions);
        Tensor loss = AutogradEngine::cross_entropy_op(logits, target_ids);

        engine.backward(loss);
        engine.clear();
        AutogradEngine::set_enabled(false);

        float loss_val = *(const float*)loss.data();
        losses.push_back(loss_val);

        optimizer.step();

        if (step % 5 == 0)
            printf("  Step %d: loss = %.4f\n", step, loss_val);
    }

    float final_loss = compute_loss(model, input_ids, positions, target_ids);
    printf("\nFinal loss: %.4f\n", final_loss);

    CHECK(final_loss < initial_loss, "Loss decreased after training");
    CHECK(losses.size() == (size_t)num_steps, "All training steps completed");
    CHECK(std::isfinite(final_loss), "Final loss is finite");

    bool monotonic = true;
    for (size_t i = 1; i < losses.size(); i++) {
        if (losses[i] > losses[0] + 0.1f) { monotonic = false; break; }
    }
    CHECK(monotonic, "Training was stable (no loss explosion)");

    printf("\n=== Results ===\n");
    printf("Initial loss: %.4f\n", initial_loss);
    printf("Final loss:   %.4f\n", final_loss);
    printf("Reduction:    %.1f%%\n",
           initial_loss > 0 ? (1.0f - final_loss / initial_loss) * 100.0f : 0.0f);

    if (g_failures == 0) { printf("\nALL TESTS PASSED\n"); return 0; }
    else { printf("\n%d TESTS FAILED\n", g_failures); return 1; }
}
