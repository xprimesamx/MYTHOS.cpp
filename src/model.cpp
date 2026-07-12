#include "oil/model.h"
#include "oil/transformer.h"
#include "oil/math.h"
#include <cstring>
#include <unordered_map>

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
    int64_t B = input_ids.dim(0);
    int64_t S = input_ids.dim(1);

    Tensor h = tok_embeddings->forward(input_ids.reshape(Shape{B * S}));
    h = h.reshape(Shape{B, S, config.hidden_size});

    Tensor causal_mask(Shape{1, 1, S, config.max_seq_len});
    float* md = causal_mask.data<float>();
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

// ========================================================================
// REAL save: write all named weight tensors to OIL format
// ========================================================================

static std::vector<std::pair<std::string, Tensor>> collect_named_tensors(const DenseModel& m) {
    std::vector<std::pair<std::string, Tensor>> tensors;
    tensors.emplace_back("tok_embeddings.weight", m.tok_embeddings->weight);
    for (size_t i = 0; i < m.layers.size(); i++) {
        std::string p = "layers." + std::to_string(i) + ".";
        const auto& l = *m.layers[i];
        tensors.emplace_back(p + "attention_norm.weight", l.attention_norm.weight);
        tensors.emplace_back(p + "ffn_norm.weight", l.ffn_norm.weight);
        tensors.emplace_back(p + "attention.q_proj.weight", l.attention.q_proj.weight);
        if (l.attention.q_proj.bias.numel() > 0)
            tensors.emplace_back(p + "attention.q_proj.bias", l.attention.q_proj.bias);
        tensors.emplace_back(p + "attention.k_proj.weight", l.attention.k_proj.weight);
        tensors.emplace_back(p + "attention.v_proj.weight", l.attention.v_proj.weight);
        tensors.emplace_back(p + "attention.o_proj.weight", l.attention.o_proj.weight);
        tensors.emplace_back(p + "ffn.gate_proj.weight", l.ffn.gate_proj.weight);
        tensors.emplace_back(p + "ffn.up_proj.weight", l.ffn.up_proj.weight);
        tensors.emplace_back(p + "ffn.down_proj.weight", l.ffn.down_proj.weight);
    }
    tensors.emplace_back("norm.weight", m.norm->weight);
    tensors.emplace_back("lm_head.weight", m.lm_head->weight);
    if (m.lm_head->bias.numel() > 0)
        tensors.emplace_back("lm_head.bias", m.lm_head->bias);
    return tensors;
}

void DenseModel::save(const std::string& oil_path) const {
    auto named = collect_named_tensors(*this);

    // First pass: build format table, tensor table, and block data in memory
    std::vector<FormatBlockEntry> ft;
    std::vector<TensorEntry> te;
    std::vector<std::string> names;
    std::vector<BlockData> blocks;
    uint32_t block_id = 0;
    uint32_t block_start = 0;

    for (auto& [name, t] : named) {
        names.push_back(name);
        int64_t numel = t.numel();
        if (numel == 0) continue;

        const float* td = t.data<float>();
        uint32_t block_count = 0;
        uint32_t bs = block_start;

        for (int64_t offset = 0; offset < numel; ) {
            uint32_t blk_size = (uint32_t)std::min((int64_t)32768, numel - offset);
            BlockData bd;
            bd.format = Format::FP32;
            bd.num_weights = blk_size;
            bd.indices.resize(blk_size * 4);
            std::memcpy(bd.indices.data(), td + offset, blk_size * 4);
            blocks.push_back(bd);

            FormatBlockEntry fbe;
            fbe.block_id = block_id++;
            fbe.format = 5; // FP32
            fbe.cb_bytes = 0;
            ft.push_back(fbe);
            block_count++;
            offset += blk_size;
        }

        TensorEntry entry;
        entry.block_start = bs;
        entry.num_blocks = block_count;
        te.push_back(entry);
        block_start = block_id;
    }

    // Write: header → format table → tensor table → block data
    OILWriter writer(oil_path);
    OILHeader hdr;
    std::memcpy(hdr.magic, "OIL1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = sizeof(TransformerConfig);
    writer.write_header(hdr, (const uint8_t*)&config);
    writer.write_format_table(ft);
    writer.write_tensor_table(te, names);
    for (auto& bd : blocks) writer.write_block(bd);
    writer.close();
}

// ========================================================================
// REAL load: read named tensors from OIL format and assign to model
// ========================================================================

void DenseModel::load(const std::string& oil_path) {
    OILReader reader(oil_path);
    if (reader.header().config_size >= sizeof(TransformerConfig)) {
        auto cfg_data = reader.read_config();
        if (cfg_data.size() >= sizeof(TransformerConfig)) {
            std::memcpy(&config, cfg_data.data(), sizeof(TransformerConfig));
        }
    }
    build_layers();

    // Collect weight map from reader
    auto tensor_names = reader.tensor_names();
    std::unordered_map<std::string, Tensor> loaded_weights;
    for (const auto& n : tensor_names) {
        loaded_weights[n] = reader.read_tensor(n);
    }

    // Build weight assign map
    auto assign = [&](const std::string& name, Tensor& dst) {
        auto it = loaded_weights.find(name);
        if (it != loaded_weights.end() && it->second.numel() == dst.numel()) {
            dst.copy_from(it->second);
        }
    };

    assign("tok_embeddings.weight", tok_embeddings->weight);
    for (size_t i = 0; i < layers.size(); i++) {
        std::string p = "layers." + std::to_string(i) + ".";
        auto& l = *layers[i];
        assign(p + "attention_norm.weight", l.attention_norm.weight);
        assign(p + "ffn_norm.weight", l.ffn_norm.weight);
        assign(p + "attention.q_proj.weight", l.attention.q_proj.weight);
        assign(p + "attention.k_proj.weight", l.attention.k_proj.weight);
        assign(p + "attention.v_proj.weight", l.attention.v_proj.weight);
        assign(p + "attention.o_proj.weight", l.attention.o_proj.weight);
        assign(p + "ffn.gate_proj.weight", l.ffn.gate_proj.weight);
        assign(p + "ffn.up_proj.weight", l.ffn.up_proj.weight);
        assign(p + "ffn.down_proj.weight", l.ffn.down_proj.weight);
    }
    assign("norm.weight", norm->weight);
    assign("lm_head.weight", lm_head->weight);
}

void Model::load(const std::string& oil_path) {
    OILReader reader(oil_path);
    if (reader.header().config_size >= sizeof(TransformerConfig)) {
        auto cfg_data = reader.read_config();
        if (cfg_data.size() >= sizeof(TransformerConfig)) {
            std::memcpy(&config, cfg_data.data(), sizeof(TransformerConfig));
        }
    }
}

void Model::save(const std::string& oil_path) const {
    OILWriter writer(oil_path);
    OILHeader hdr;
    std::memcpy(hdr.magic, "OIL1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = sizeof(TransformerConfig);
    writer.write_header(hdr, (const uint8_t*)&config);
    writer.close();
}

} // namespace oil
