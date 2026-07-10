#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/model.h"
#include "oil/optimizer.h"
#include "oil/tokenizer.h"
#include "oil/autograd.h"
#include <vector>
#include <string>
#include <functional>

namespace oil {

struct TrainConfig {
    int64_t batch_size = 8;
    int64_t seq_length = 512;
    int num_epochs = 3;
    float learning_rate = 3e-4f;
    float weight_decay = 1e-2f;
    int warmup_steps = 100;
    int log_interval = 10;
    int save_interval = 1000;
    std::string output_path = "model.oil";
    bool use_ste = true;  // OIL-native training
};

class DataLoader {
public:
    DataLoader(Tokenizer* tokenizer, const std::string& data_path,
               int64_t batch_size, int64_t seq_length);
    
    bool next_batch(Tensor& input_ids, Tensor& labels);
    void shuffle();
    void reset();
    int64_t num_batches() const;
    
private:
    Tokenizer* tokenizer_;
    std::vector<int> tokenized_data_;
    int64_t batch_size_;
    int64_t seq_length_;
    int64_t current_pos_ = 0;
    int64_t num_batches_ = 0;
};

struct TrainMetrics {
    float loss = 0;
    float perplexity = 0;
    float learning_rate = 0;
    int tokens_per_sec = 0;
    int step = 0;
};

class Trainer {
public:
    Trainer(Model* model, Tokenizer* tokenizer);
    
    void compile(AdamW* optimizer);
    void fit(DataLoader& dataloader, const TrainConfig& cfg);
    
    float train_step(const Tensor& input_ids, const Tensor& labels);
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    
    // Callbacks
    using LogCallback = std::function<void(const TrainMetrics&)>;
    void set_log_callback(LogCallback cb);
    
    const TrainMetrics& metrics() const;
    
private:
    Model* model_;
    Tokenizer* tokenizer_;
    AdamW* optimizer_ = nullptr;
    TrainMetrics metrics_;
    LogCallback log_cb_;
    
    int step_ = 0;
};

} // namespace oil
