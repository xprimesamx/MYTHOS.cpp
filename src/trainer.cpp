#include "oil/trainer.h"
#include "oil/math.h"
#include "oil/autograd.h"
#include "oil/optimizer.h"
#include "oil/transformer.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>

namespace oil {

static void collect_trainer_params(DenseModel* dm, std::vector<Tensor*>& params) {
    if (!dm) return;
    params.push_back(&dm->tok_embeddings->weight);
    for (auto& layer : dm->layers) {
        params.push_back(&layer->attention_norm.weight);
        params.push_back(&layer->attention.q_proj.weight);
        params.push_back(&layer->attention.q_proj.bias);
        params.push_back(&layer->attention.k_proj.weight);
        params.push_back(&layer->attention.k_proj.bias);
        params.push_back(&layer->attention.v_proj.weight);
        params.push_back(&layer->attention.v_proj.bias);
        params.push_back(&layer->attention.o_proj.weight);
        params.push_back(&layer->attention.o_proj.bias);
        params.push_back(&layer->ffn_norm.weight);
        params.push_back(&layer->ffn.gate_proj.weight);
        params.push_back(&layer->ffn.gate_proj.bias);
        params.push_back(&layer->ffn.up_proj.weight);
        params.push_back(&layer->ffn.up_proj.bias);
        params.push_back(&layer->ffn.down_proj.weight);
        params.push_back(&layer->ffn.down_proj.bias);
    }
    params.push_back(&dm->norm->weight);
    params.push_back(&dm->lm_head->weight);
    params.push_back(&dm->lm_head->bias);
}

// ===========================================================================
// DataLoader
// ===========================================================================

DataLoader::DataLoader(Tokenizer* tokenizer, const std::string& data_path,
                       int64_t batch_size, int64_t seq_length)
    : tokenizer_(tokenizer), batch_size_(batch_size), seq_length_(seq_length), current_pos_(0) {
    std::ifstream f(data_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        num_batches_ = 0;
        return;
    }
    size_t size = (size_t)f.tellg();
    f.seekg(0);
    std::string text((size_t)size, '\0');
    f.read(&text[0], size);
    tokenized_data_ = tokenizer_->encode(text);
    int64_t total_tokens = (int64_t)tokenized_data_.size();
    int64_t tokens_per_batch = batch_size_ * seq_length_;
    num_batches_ = total_tokens / tokens_per_batch;
}

bool DataLoader::next_batch(Tensor& input_ids, Tensor& labels) {
    if (current_pos_ >= num_batches_) return false;

    int64_t start = current_pos_ * batch_size_ * seq_length_;
    int64_t end = start + batch_size_ * seq_length_;
    if (end + 1 > (int64_t)tokenized_data_.size()) return false;

    float* id = (float*)input_ids.data();
    float* ld = (float*)labels.data();

    for (int64_t i = 0; i < batch_size_ * seq_length_; i++) {
        int64_t src_idx = start + i;
        int64_t label_idx = src_idx + 1;
        if (label_idx >= (int64_t)tokenized_data_.size()) {
            label_idx = (int64_t)tokenized_data_.size() - 1;
        }
        id[i] = (float)tokenized_data_[src_idx];
        ld[i] = (float)tokenized_data_[label_idx];
    }

    current_pos_++;
    return true;
}

void DataLoader::shuffle() {
    if (tokenized_data_.empty()) return;
    static std::mt19937 g(42);
    std::shuffle(tokenized_data_.begin(), tokenized_data_.end(), g);
    current_pos_ = 0;
}

void DataLoader::reset() {
    current_pos_ = 0;
}

int64_t DataLoader::num_batches() const {
    return num_batches_;
}

// ===========================================================================
// Trainer
// ===========================================================================

Trainer::Trainer(Model* m, Tokenizer* t) : model_(m), tokenizer_(t), step_(0) {}

void Trainer::compile(AdamW* opt) {
    optimizer_ = opt;
    DenseModel* dm = dynamic_cast<DenseModel*>(model_);
    if (dm) {
        model_params_.clear();
        collect_trainer_params(dm, model_params_);
        auto& engine = AutogradEngine::instance();
        for (auto* p : model_params_) {
            p->requires_grad(true);
            engine.register_parameter(p);
        }
        optimizer_->add_param_group(model_params_);
    }
}

void Trainer::fit(DataLoader& dl, const TrainConfig& cfg) {
    Tensor input_ids(Shape{cfg.batch_size, cfg.seq_length}, DType::F32);
    Tensor labels(Shape{cfg.batch_size, cfg.seq_length}, DType::F32);

    for (int epoch = 0; epoch < cfg.num_epochs; epoch++) {
        dl.reset();
        while (dl.next_batch(input_ids, labels)) {
            float loss = train_step(input_ids, labels);

            metrics_.loss = loss;
            metrics_.perplexity = std::exp(loss);
            metrics_.learning_rate = optimizer_ ? optimizer_->get_lr() : cfg.learning_rate;
            metrics_.step = step_;

            if (step_ % cfg.log_interval == 0 && log_cb_) {
                log_cb_(metrics_);
            }
            if (step_ % cfg.save_interval == 0 && cfg.save_interval > 0) {
                save_checkpoint(cfg.output_path);
            }
            step_++;
        }
    }
    save_checkpoint(cfg.output_path);
}

float Trainer::train_step(const Tensor& input_ids, const Tensor& labels) {
    if (!optimizer_) return 0;

    int64_t B = input_ids.dim(0);
    int64_t S = input_ids.dim(1);

    Tensor positions(Shape{B, S}, DType::F32);
    float* pd = positions.data<float>();
    for (int64_t i = 0; i < B * S; i++)
        pd[i] = (float)(i % S);

    optimizer_->zero_grad();

    AutogradEngine::set_enabled(true);
    // Re-register parameters (clear() removes them from param_map_)
    auto& engine = AutogradEngine::instance();
    for (auto* p : model_params_) {
        engine.register_parameter(p);
    }
    Tensor logits = model_->forward(input_ids, positions);
    Tensor loss = AutogradEngine::cross_entropy_op(logits, labels);
    engine.backward(loss);
    engine.clear();
    AutogradEngine::set_enabled(false);
    optimizer_->step();

    return *(const float*)loss.data();
}

void Trainer::save_checkpoint(const std::string& path) {
    model_->save(path);
}

void Trainer::load_checkpoint(const std::string& path) {
    model_->load(path);
}

void Trainer::set_log_callback(LogCallback cb) {
    log_cb_ = cb;
}

const TrainMetrics& Trainer::metrics() const {
    return metrics_;
}

} // namespace oil
