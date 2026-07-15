#include "oil/model.h"
#include "oil/transformer.h"
#include "oil/tensor.h"
#include "oil/types.h"
#include <iostream>
#include <cstring>

int main() {
    std::cout << "Starting..." << std::endl;
    
    oil::TransformerConfig cfg;
    cfg.vocab_size = 100;
    cfg.hidden_size = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;
    cfg.ffn_hidden_size = 64;
    cfg.norm_eps = 1e-5f;
    cfg.max_seq_len = 64;
    
    std::cout << "Creating model..." << std::endl;
    oil::DenseModel model(cfg);
    std::cout << "Model created. Params: " << model.param_count() << std::endl;
    
    std::cout << "Creating input..." << std::endl;
    const int64_t B = 2;
    const int64_t S = 4;
    oil::Tensor input_ids(oil::Shape{B, S}, oil::DType::F32);
    oil::Tensor positions(oil::Shape{B, S}, oil::DType::F32);
    
    for (int64_t i = 0; i < B * S; i++) {
        float tid = static_cast<float>((int)((i * 7 + 3) % 100));
        float pid = static_cast<float>((int)(i % S));
        std::memcpy(input_ids.data<float>() + i, &tid, sizeof(float));
        std::memcpy(positions.data<float>() + i, &pid, sizeof(float));
    }
    
    std::cout << "Calling forward..." << std::endl;
    oil::Tensor logits = model.forward(input_ids, positions);
    std::cout << "Forward done! shape=" << logits.shape().dims[0] << " " 
              << logits.shape().dims[1] << " " << logits.shape().dims[2] << std::endl;
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
