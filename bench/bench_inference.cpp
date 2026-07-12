#include "oil/model.h"
#include "oil/transformer.h"
#include "oil/tokenizer.h"
#include "oil/generator.h"
#include "oil/sampler.h"
#include "oil/tensor.h"
#include "oil/types.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

// NOTE: For real inference speed measurements, load a trained model checkpoint:
//   oil::Tokenizer tokenizer("tokenizer.bin");
//   oil::DenseModel model("model.oil");
//   oil::Generator generator(&model, &tokenizer);
//   auto t0 = now_sec();
//   std::string output = generator.generate("prompt", 256);
//   double dt = now_sec() - t0;
//   double tok_s = 256.0 / dt;
// This file benchmarks synthetic random models for forward-pass latency.

int main() {
    std::cout << "=== OIL Inference Benchmarks ===" << std::endl;

    // Prefill benchmark with synthetic model
    std::vector<std::pair<int, int>> model_sizes = {
        {768, 12},   // ~110M
        {512, 8},    // ~35M
        {256, 4},    // ~4M
    };

    for (auto [hidden, layers] : model_sizes) {
        oil::TransformerConfig cfg;
        cfg.vocab_size = 32000;
        cfg.hidden_size = hidden;
        cfg.num_layers = layers;
        cfg.num_heads = hidden / 64;
        cfg.head_dim = 64;
        cfg.ffn_hidden_size = hidden * 4;
        cfg.max_seq_len = 2048;
        cfg.norm_eps = 1e-5f;

        oil::DenseModel model(cfg);
        std::cout << "\nModel: hidden=" << hidden << " layers=" << layers
                  << " params=" << model.param_count() << std::endl;

        const int64_t B = 1;
        std::vector<int64_t> seq_lens = {16, 32, 64, 128};

        for (int64_t S : seq_lens) {
            if (S > cfg.max_seq_len) continue;

            oil::Tensor input_ids(oil::Shape{B, S}, oil::DType::F32);
            oil::Tensor positions(oil::Shape{B, S}, oil::DType::F32);

            for (int64_t i = 0; i < S; i++) {
                ((int*)input_ids.data())[i] = (int)((i * 7) % cfg.vocab_size);
                ((int*)positions.data())[i] = (int)i;
            }

            // Warmup
            for (int i = 0; i < 5; i++)
                model.forward(input_ids, positions);

            int iters = 20;
            double t0 = now_sec();
            for (int i = 0; i < iters; i++)
                model.forward(input_ids, positions);
            double dt = (now_sec() - t0) / iters;

            double tokens_per_sec = S / dt;
            std::cout << "  S=" << std::setw(4) << S
                      << " | avg: " << std::fixed << std::setprecision(4) << (dt * 1000) << " ms"
                      << " | " << std::fixed << std::setprecision(0) << tokens_per_sec << " tok/s" << std::endl;
        }
    }

    // Decode (single-token) benchmark
    std::cout << "\n=== Single-Token Decode Latency ===" << std::endl;
    for (auto [hidden, layers] : model_sizes) {
        oil::TransformerConfig cfg;
        cfg.vocab_size = 32000;
        cfg.hidden_size = hidden;
        cfg.num_layers = layers;
        cfg.num_heads = hidden / 64;
        cfg.head_dim = 64;
        cfg.ffn_hidden_size = hidden * 4;
        cfg.max_seq_len = 2048;

        oil::DenseModel model(cfg);

        oil::Tensor single_in(oil::Shape{1, 1}, oil::DType::F32);
        oil::Tensor single_pos(oil::Shape{1, 1}, oil::DType::F32);
        ((int*)single_in.data())[0] = 0;
        ((int*)single_pos.data())[0] = 0;

        // Warmup
        for (int i = 0; i < 20; i++)
            model.forward(single_in, single_pos);

        int iters = 100;
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            model.forward(single_in, single_pos);
        double dt = (now_sec() - t0) / iters;

        std::cout << "hidden=" << hidden << " layers=" << layers
                  << ": " << std::fixed << std::setprecision(3) << (dt * 1000) << " ms/token"
                  << " (" << std::fixed << std::setprecision(0) << (1.0 / dt) << " tok/s)" << std::endl;
    }

    return 0;
}
