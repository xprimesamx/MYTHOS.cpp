#include "oil/model.h"
#include "oil/transformer.h"
#include "oil/math.h"
#include <cstring>

namespace oil {

DenseModel::DenseModel(const TransformerConfig& cfg) {
    config = cfg;
    build_layers();
}

void DenseModel::build_layers() {
    tok_embeddings = std::make_unique<Embedding>(config.vocab_size, config.hidden_size);
    layers.clear();
    for (int64_t i = 0; i < config.num_layers; i++) {
        layers.push_back(std::make_unique<TransformerBlock>(config));
    }
    norm = std::make_unique<RMSNorm>(config.hidden_size, config.norm_eps);
    lm_head = std::make_unique<Linear>(config.hidden_size, config.vocab_size);
}

int64_t DenseModel::vocab_size() const { return config.vocab_size; }

int64_t DenseModel::param_count() const {
    int64_t count = tok_embeddings->param_count();
    for (auto& l : layers) {
        count += l->attention.q_proj.param_count();
        count += l->attention.k_proj.param_count();
        count += l->attention.v_proj.param_count();
        count += l->attention.o_proj.param_count();
        count += l->ffn.gate_proj.param_count();
        count += l->ffn.up_proj.param_count();
        count += l->ffn.down_proj.param_count();
        count += l->attention_norm.weight.numel();
        count += l->ffn_norm.weight.numel();
    }
    count += norm->weight.numel();
    count += lm_head->param_count();
    return count;
}

Tensor DenseModel::forward(const Tensor& input_ids, const Tensor& positions) {
    int64_t B = input_ids.shape().dims[0];
    int64_t S = input_ids.shape().dims[1];
    
    Tensor h = tok_embeddings->forward(input_ids.reshape(Shape{B * S}));
    h = h.reshape(Shape{B, S, config.hidden_size});
    
    Tensor causal_mask(Shape{1, 1, S, config.max_seq_len}, DType::F32);
    // Causality: upper triangle = -inf
    float* md = (float*)causal_mask.data();
    for (int64_t s = 0; s < S; s++) {
        for (int64_t t = 0; t < config.max_seq_len; t++) {
            md[s * config.max_seq_len + t] = (t > s) ? -INFINITY : 0.0f;
        }
    }
    
    KVCache cache(config.num_layers, config.max_seq_len, config.num_heads,
                  config.head_dim);
    
    for (int64_t i = 0; i < config.num_layers; i++) {
        h = layers[i]->forward(h, positions, causal_mask, cache, (int)i);
    }
    
    h = norm->forward(h);
    return lm_head->forward(h);
}

void DenseModel::load(const std::string& oil_path) {
    OILReader reader(oil_path);
    config.vocab_size = reader.header().config_size; // simplified
    
    // Read named tensors
    for (const auto& name : reader.tensor_names()) {
        Tensor t = reader.read_tensor(name);
        // Assign by name
    }
}

void DenseModel::save(const std::string& oil_path) const {
    OILWriter writer(oil_path);
    OILHeader hdr;
    memcpy(hdr.magic, "OIL1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = sizeof(TransformerConfig);
    writer.write_header(hdr, (const uint8_t*)&config);
}

void Model::load(const std::string& oil_path) {}
void Model::save(const std::string& oil_path) const {}

} // namespace oil
