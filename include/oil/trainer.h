#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/model.h"
#include "oil/optimizer.h"
#include "oil/tokenizer.h"
#include "oil/autograd.h"
#include "oil/codebook.h"
#include <vector>
#include <string>
#include <functional>
#include <fstream>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>

namespace oil {

struct TrainConfig {
    int64_t batch_size = 8;
    int64_t seq_length = 512;
    int num_epochs = 3;
    int train_steps = 10000;
    float learning_rate = 3e-4f;
    float weight_decay = 1e-2f;
    int warmup_steps = 100;
    int log_interval = 10;
    int save_interval = 1000;
    int val_interval = 500;
    std::string output_path = "model.oil";
    bool use_ste = true;
    float max_grad_norm = 1.0f;
    int gradient_accumulation_steps = 1;
    AdamW::Schedule schedule = AdamW::Schedule::WARMUP_COSINE;
    bool mixed_precision = false;
    float loss_scale = 128.0f;
    int loss_scale_interval = 2000;
    // C17: Label smoothing
    float label_smoothing = 0.0f;
    // C19: R-Drop consistency
    bool use_rdrop = false;
    float rdrop_alpha = 1.0f;
    // C8: Data augmentation
    bool data_augmentation = false;
    float aug_noise_std = 0.01f;
    float aug_mask_prob = 0.0f;
    // C9: Curriculum learning
    bool curriculum = false;
    int curriculum_epochs = 3;
};

struct AugmentConfig {
    bool enabled = false;
    float mask_prob = 0.0f;
    float noise_std = 0.01f;
    float replace_prob = 0.0f;
};

class DataLoader {
public:
    DataLoader(Tokenizer* tokenizer, const std::string& data_path,
               int64_t batch_size, int64_t seq_length,
               bool stream_from_disk = false);
    DataLoader(Tokenizer* tokenizer, const std::string& data_path,
               int64_t batch_size, int64_t seq_length,
               bool stream_from_disk, int num_workers,
               int64_t prefetch_capacity, bool use_mmap);
    ~DataLoader();
    
    bool next_batch(Tensor& input_ids, Tensor& labels);
    void shuffle(int epoch = 0);
    void reset();
    int64_t num_batches() const;
    int64_t batch_size() const { return batch_size_; }
    int64_t seq_length() const { return seq_length_; }
    
    // C8: Data augmentation
    void set_augmentation(const AugmentConfig& cfg) { aug_cfg_ = cfg; }
    void apply_augmentation(Tensor& input_ids, Tensor& labels);
    
    // C9: Curriculum learning
    struct CurriculumState {
        int current_epoch = 0;
        int total_epochs = 3;
        float seq_len_multiplier = 0.5f;
        int64_t effective_seq_length() const {
            float t = (float)current_epoch / (float)std::max(total_epochs, 1);
            int64_t min_len = 64;
            return std::max(min_len, (int64_t)(512 * (min_len / 512.0f + t * (1.0f - min_len / 512.0f))));
        }
    };
    void set_curriculum(int total_epochs) { cur_state_.total_epochs = total_epochs; }
    void curriculum_step(int epoch) { cur_state_.current_epoch = epoch; }
    
private:
    Tokenizer* tokenizer_;
    std::vector<int> tokenized_data_;
    int64_t batch_size_;
    int64_t seq_length_;
    int64_t current_pos_ = 0;
    int64_t num_batches_ = 0;
    bool streaming_ = false;
    std::string data_path_;
    std::ifstream file_stream_;
    std::vector<int> stream_chunk_;
    int64_t stream_file_offset_ = 0;
    static constexpr int64_t STREAM_CHUNK_TOKENS = 1024 * 1024;
    
    void tokenize_chunk();
    
    // C3: Memory-mapped I/O
    bool use_mmap_ = false;
    void* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;
#ifdef _WIN32
    void* mmap_handle_ = nullptr;
    void* file_handle_ = nullptr;
#else
    int mmap_fd_ = -1;
#endif
    void open_mmap(const std::string& path);
    void close_mmap();
    
    // C2/C16: Multi-worker prefetch
    int num_workers_ = 0;
    int64_t prefetch_capacity_ = 0;
    std::thread prefetch_thread_;
    std::mutex prefetch_mutex_;
    std::queue<std::pair<Tensor, Tensor>> prefetch_queue_;
    std::atomic<bool> prefetch_running_{false};
    void prefetch_worker();
    void start_prefetch();
    void stop_prefetch();
    
    // C8: Data augmentation
    AugmentConfig aug_cfg_;
    
    // C9: Curriculum learning
    CurriculumState cur_state_;
};

struct TrainMetrics {
    float loss = 0;
    float perplexity = 0;
    float val_loss = 0;
    float val_perplexity = 0;
    float learning_rate = 0;
    float grad_norm = 0;
    int tokens_per_sec = 0;
    int step = 0;
    int epoch = 0;
};

class Trainer {
public:
    Trainer(Model* model, Tokenizer* tokenizer);
    
    void compile(AdamW* optimizer, const TrainConfig& cfg = TrainConfig{});
    void fit(DataLoader& train_dl, const TrainConfig& cfg,
             DataLoader* val_dl = nullptr);
    
    float train_step(const Tensor& input_ids, const Tensor& labels);
    float micro_step(const Tensor& input_ids, const Tensor& labels, float loss_scale = 1.0f);
    float eval_loss(DataLoader& val_dl, int64_t max_batches = 20);
    float clip_gradients(float max_norm);
    void unscale_gradients(float scale);
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    
    // Callbacks
    using LogCallback = std::function<void(const TrainMetrics&)>;
    using EpochCallback = std::function<void(int epoch, const TrainMetrics&)>;
    using StepCallback = std::function<void(int step, const TrainMetrics&)>;
    void set_log_callback(LogCallback cb);
    void set_epoch_callback(EpochCallback cb);
    void set_step_callback(StepCallback cb);
    
    const TrainMetrics& metrics() const;
    const std::vector<Tensor*>& get_model_params() const { return model_params_; }
    std::vector<Codebook*> get_codebooks() const { return codebooks_; }
    void set_codebooks(const std::vector<Codebook*>& cbs) { codebooks_ = cbs; }
    
    // C20: EMA tracking
    void ema_init(float decay = 0.999f);
    void ema_step();
    void ema_apply();
    void ema_swap();
    
    // C14: Enhanced mixed precision
    void dynamic_loss_scale(float grad_norm, float max_grad_norm);
    
    // C17: Label smoothing cross-entropy
    Tensor label_smoothing_loss(const Tensor& logits, const Tensor& labels, float smoothing);
    // C19: R-Drop consistency loss
    float rdrop_loss(const Tensor& input_ids, const Tensor& labels, float alpha);
    
private:
    Model* model_;
    Tokenizer* tokenizer_;
    AdamW* optimizer_ = nullptr;
    TrainMetrics metrics_;
    LogCallback log_cb_;
    EpochCallback epoch_cb_;
    StepCallback step_cb_;
    std::vector<Tensor*> model_params_;
    std::vector<Codebook*> codebooks_;
    
    int step_ = 0;
    float loss_scale_ = 1.0f;
    int loss_scale_interval_ = 2000;
    int steps_since_scale_update_ = 0;
    
    // C20: EMA state
    std::vector<Tensor> ema_params_;
    float ema_decay_ = 0.999f;
    bool ema_enabled_ = false;
};

} // namespace oil
