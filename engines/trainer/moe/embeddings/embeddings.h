#pragma once
#include "oil/tensor.h"

namespace oil {
namespace multimodal {

class SentenceEmbedder {
public:
    SentenceEmbedder(int64_t hidden_size, int64_t output_size);
    Tensor forward(const Tensor& token_embeddings);

    Tensor projection;
    int64_t hidden_size;
    int64_t output_size;
};

} // namespace multimodal
} // namespace oil
