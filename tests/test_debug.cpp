#include "oil/model.h"
#include "oil/transformer.h"
#include "oil/tensor.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    oil::TransformerConfig cfg;
    cfg.vocab_size = 100;
    cfg.hidden_size = 32;
    cfg.num_layers = 2;  // match test_model
    cfg.num_heads = 4;
    cfg.head_dim = 8;
    cfg.ffn_hidden_size = 64;
    cfg.max_seq_len = 64;

    oil::DenseModel model(cfg);
    const int64_t B = 2, S = 4;

    oil::Tensor input_ids(oil::Shape{B, S}, oil::DType::F32);
    oil::Tensor positions(oil::Shape{B, S}, oil::DType::F32);
    unsigned int seed = 42;
    for (int64_t i = 0; i < B*S; i++) {
        float tid = static_cast<float>((int)((seed >> 16) % cfg.vocab_size));
        float pid = static_cast<float>((int)(i % S));
        std::memcpy(input_ids.data<float>() + i, &tid, sizeof(float));
        std::memcpy(positions.data<float>() + i, &pid, sizeof(float));
        seed = seed * 1103515245u + 12345u;
    }
    std::fprintf(stderr, "Input ready, calling forward\n"); std::fflush(stderr);
    oil::Tensor logits = model.forward(input_ids, positions);
    std::fprintf(stderr, "Forward done! shape=%lld %lld %lld\n",
        (long long)logits.dim(0), (long long)logits.dim(1), (long long)logits.dim(2));
    std::fflush(stderr);
    return 0;
}
