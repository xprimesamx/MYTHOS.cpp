#define NOMINMAX
#include "oil/trainer.h"
#include "oil/math.h"
#include "oil/autograd.h"
#include "oil/optimizer.h"
#include "oil/transformer.h"
#include "oil/flash_attention.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace oil {

// FP16 encode/decode
static inline uint16_t float_to_fp16(float v) {
    uint32_t u32;
    memcpy(&u32, &v, sizeof(u32));
    uint32_t sign = (u32 >> 31) & 1;
    uint32_t exp = (u32 >> 23) & 0xFF;
    uint32_t mant = u32 & 0x7FFFFF;
    if (exp == 0) return (uint16_t)(sign << 15);
    if (exp >= 0x8E) return (uint16_t)((sign << 15) | 0x7C00 | (mant ? 0x200 : 0));
    uint16_t f16 = (uint16_t)((sign << 15) | ((exp - 112) << 10) | (mant >> 13));
    return f16;
}

static inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        float v = (float)mant * 0.000000059604644775390625f;
        return sign ? -v : v;
    }
    if (exp == 31) {
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    }
    uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

static void quantize_params_to_fp16(const std::vector<Tensor*>& params) {
    for (auto* p : params) {
        float* d = p->data<float>();
        int64_t n = p->numel();
        for (int64_t i = 0; i < n; i++) {
            uint16_t f16 = float_to_fp16(d[i]);
            d[i] = fp16_to_float(f16);
        }
    }
}

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
                       int64_t batch_size, int64_t seq_length, bool stream_from_disk)
    : tokenizer_(tokenizer), batch_size_(batch_size), seq_length_(seq_length),
      current_pos_(0), streaming_(stream_from_disk), data_path_(data_path) {
    if (streaming_) {
        file_stream_.open(data_path, std::ios::binary);
        if (!file_stream_.is_open()) {
            num_batches_ = 0;
            return;
        }
        file_stream_.seekg(0, std::ios::end);
        int64_t file_size = (int64_t)file_stream_.tellg();
        file_stream_.seekg(0);
        int64_t approx_tokens = file_size / 4;
        int64_t tokens_per_batch = batch_size_ * seq_length_;
        num_batches_ = (std::max)(approx_tokens / tokens_per_batch, (int64_t)1);
        tokenize_chunk();
    } else {
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
}

DataLoader::DataLoader(Tokenizer* tokenizer, const std::string& data_path,
                       int64_t batch_size, int64_t seq_length,
                       bool stream_from_disk, int num_workers,
                       int64_t prefetch_capacity, bool use_mmap)
    : tokenizer_(tokenizer), batch_size_(batch_size), seq_length_(seq_length),
      current_pos_(0), streaming_(stream_from_disk), data_path_(data_path),
      num_workers_(num_workers), prefetch_capacity_(prefetch_capacity),
      use_mmap_(use_mmap) {
    if (use_mmap_) {
        open_mmap(data_path_);
        if (mmap_ptr_) {
            std::string text((const char*)mmap_ptr_, mmap_size_);
            tokenized_data_ = tokenizer_->encode(text);
            int64_t total_tokens = (int64_t)tokenized_data_.size();
            int64_t tokens_per_batch = batch_size_ * seq_length_;
            num_batches_ = total_tokens / tokens_per_batch;
        }
    } else if (streaming_) {
        file_stream_.open(data_path, std::ios::binary);
        if (!file_stream_.is_open()) { num_batches_ = 0; return; }
        file_stream_.seekg(0, std::ios::end);
        int64_t file_size = (int64_t)file_stream_.tellg();
        file_stream_.seekg(0);
        int64_t approx_tokens = file_size / 4;
        int64_t tokens_per_batch = batch_size_ * seq_length_;
        num_batches_ = (std::max)(approx_tokens / tokens_per_batch, (int64_t)1);
        tokenize_chunk();
    } else {
        std::ifstream f(data_path, std::ios::binary | std::ios::ate);
        if (!f.is_open()) { num_batches_ = 0; return; }
        size_t size = (size_t)f.tellg();
        f.seekg(0);
        std::string text((size_t)size, '\0');
        f.read(&text[0], size);
        tokenized_data_ = tokenizer_->encode(text);
        int64_t total_tokens = (int64_t)tokenized_data_.size();
        int64_t tokens_per_batch = batch_size_ * seq_length_;
        num_batches_ = total_tokens / tokens_per_batch;
    }
    if (num_workers_ > 0 && prefetch_capacity_ > 0)
        start_prefetch();
}

// C3: Memory-mapped I/O
void DataLoader::open_mmap(const std::string& path) {
#ifdef _WIN32
    file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file_handle_ == INVALID_HANDLE_VALUE) return;
    LARGE_INTEGER li;
    GetFileSizeEx(file_handle_, &li);
    mmap_size_ = (size_t)li.QuadPart;
    mmap_handle_ = CreateFileMappingA(file_handle_, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mmap_handle_) { CloseHandle(file_handle_); file_handle_ = nullptr; return; }
    mmap_ptr_ = MapViewOfFile(mmap_handle_, FILE_MAP_READ, 0, 0, 0);
#else
    mmap_fd_ = open(path.c_str(), O_RDONLY);
    if (mmap_fd_ < 0) return;
    struct stat st;
    fstat(mmap_fd_, &st);
    mmap_size_ = (size_t)st.st_size;
    mmap_ptr_ = mmap(NULL, mmap_size_, PROT_READ, MAP_PRIVATE, mmap_fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) { mmap_ptr_ = nullptr; close(mmap_fd_); mmap_fd_ = -1; }
#endif
}

void DataLoader::close_mmap() {
    if (!mmap_ptr_) return;
#ifdef _WIN32
    UnmapViewOfFile(mmap_ptr_);
    if (mmap_handle_) CloseHandle(mmap_handle_);
    if (file_handle_) CloseHandle(file_handle_);
#else
    munmap(mmap_ptr_, mmap_size_);
    if (mmap_fd_ >= 0) close(mmap_fd_);
#endif
    mmap_ptr_ = nullptr;
    mmap_size_ = 0;
}

DataLoader::~DataLoader() {
    stop_prefetch();
    close_mmap();
    if (file_stream_.is_open()) file_stream_.close();
}

// C2/C16: Multi-worker prefetch
void DataLoader::prefetch_worker() {
    while (prefetch_running_) {
        Tensor input_ids(Shape{batch_size_, seq_length_}, DType::F32);
        Tensor labels(Shape{batch_size_, seq_length_}, DType::F32);
        bool ok;
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            ok = next_batch(input_ids, labels);
        }
        if (!ok) break;
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            if (prefetch_queue_.size() < (size_t)prefetch_capacity_)
                prefetch_queue_.push({input_ids.clone(), labels.clone()});
        }
    }
}

void DataLoader::start_prefetch() {
    prefetch_running_ = true;
    prefetch_thread_ = std::thread(&DataLoader::prefetch_worker, this);
}

void DataLoader::stop_prefetch() {
    prefetch_running_ = false;
    if (prefetch_thread_.joinable()) prefetch_thread_.join();
}

// C8: Data augmentation
void DataLoader::apply_augmentation(Tensor& input_ids, Tensor& labels) {
    if (!aug_cfg_.enabled) return;
    float* id = input_ids.data<float>();
    float* ld = labels.data<float>();
    int64_t n = input_ids.numel();
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, aug_cfg_.noise_std);
    for (int64_t i = 0; i < n; i++) {
        if (aug_cfg_.mask_prob > 0 && dist(rng) < aug_cfg_.mask_prob)
            id[i] = 0;
        if (aug_cfg_.replace_prob > 0 && dist(rng) < aug_cfg_.replace_prob)
            id[i] = (float)(rng() % 32000);
        if (aug_cfg_.noise_std > 0)
            id[i] += noise(rng);
        if (aug_cfg_.mask_prob > 0 && dist(rng) < aug_cfg_.mask_prob)
            ld[i] = 0;
    }
}



void DataLoader::tokenize_chunk() {
    if (!file_stream_.is_open()) return;
    stream_chunk_.clear();
    std::string buffer;
    buffer.resize(STREAM_CHUNK_TOKENS * 4);
    file_stream_.read(&buffer[0], STREAM_CHUNK_TOKENS * 4);
    size_t bytes_read = (size_t)file_stream_.gcount();
    if (bytes_read == 0) return;
    buffer.resize(bytes_read);
    tokenized_data_ = tokenizer_->encode(buffer);
    stream_file_offset_ = 0;
}

bool DataLoader::next_batch(Tensor& input_ids, Tensor& labels) {
    if (current_pos_ >= num_batches_) return false;

    if (streaming_) {
        int64_t tokens_needed = batch_size_ * seq_length_;
        while ((int64_t)(stream_chunk_.size()) - stream_file_offset_ < tokens_needed + 1) {
            if (file_stream_.eof()) return false;
            tokenize_chunk();
            if (tokenized_data_.empty()) return false;
            stream_chunk_.insert(stream_chunk_.end(),
                                 tokenized_data_.begin(), tokenized_data_.end());
        }
        float* id = (float*)input_ids.data();
        float* ld = (float*)labels.data();
        for (int64_t i = 0; i < tokens_needed; i++) {
            id[i] = (float)stream_chunk_[stream_file_offset_ + i];
            ld[i] = (float)stream_chunk_[stream_file_offset_ + i + 1];
        }
        stream_file_offset_ += tokens_needed;
        current_pos_++;
        return true;
    }

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

void DataLoader::shuffle(int epoch) {
    if (tokenized_data_.empty()) return;
    std::mt19937 g(42 + epoch);
    std::shuffle(tokenized_data_.begin(), tokenized_data_.end(), g);
    current_pos_ = 0;
}

void DataLoader::reset() {
    current_pos_ = 0;
    if (streaming_ && file_stream_.is_open()) {
        file_stream_.clear();
        file_stream_.seekg(0);
        stream_chunk_.clear();
        stream_file_offset_ = 0;
        tokenize_chunk();
    }
}

int64_t DataLoader::num_batches() const {
    return num_batches_;
}

// ===========================================================================
// StreamingDataLoader (Task 118)
// ===========================================================================

StreamingDataLoader::StreamingDataLoader(Tokenizer* tokenizer, const std::string& data_path,
                                         int64_t batch_size, int64_t seq_length)
    : tokenizer_(tokenizer), batch_size_(batch_size), seq_length_(seq_length) {
    file_.open(data_path, std::ios::binary);
    if (!file_.is_open()) { num_batches_ = 0; return; }
    file_.seekg(0, std::ios::end);
    int64_t file_size = (int64_t)file_.tellg();
    file_.seekg(0);
    int64_t approx_tokens = file_size / 4;     // ~4 bytes per token estimate
    int64_t tokens_per_batch = batch_size_ * seq_length_;
    num_batches_ = (std::max)(approx_tokens / tokens_per_batch, (int64_t)1);
    fill_buffer();
}

void StreamingDataLoader::fill_buffer() {
    if (eof_) return;
    std::string chunk;
    chunk.resize(chunk_bytes_);
    file_.read(&chunk[0], chunk_bytes_);
    size_t got = (size_t)file_.gcount();
    if (got == 0) { eof_ = true; return; }
    chunk.resize(got);
    auto toks = tokenizer_->encode(chunk);
    buffer_.insert(buffer_.end(), toks.begin(), toks.end());
}

bool StreamingDataLoader::next_batch(Tensor& input_ids, Tensor& labels) {
    int64_t need = batch_size_ * seq_length_;
    while ((int64_t)buffer_.size() < need + 1) {
        if (eof_) return false;
        fill_buffer();
        if (eof_ && (int64_t)buffer_.size() < need + 1) return false;
    }
    float* id = input_ids.data<float>();
    float* ld = labels.data<float>();
    for (int64_t i = 0; i < need; i++) {
        id[i] = (float)buffer_[i];
        ld[i] = (float)buffer_[i + 1];
    }
    buffer_.erase(buffer_.begin(), buffer_.begin() + need);
    return true;
}

void StreamingDataLoader::reset() {
    if (file_.is_open()) {
        file_.clear();
        file_.seekg(0);
    }
    buffer_.clear();
    eof_ = false;
    fill_buffer();
}

// ===========================================================================
// Trainer
// ===========================================================================

Trainer::Trainer(Model* m, Tokenizer* t) : model_(m), tokenizer_(t), step_(0) {}

void Trainer::compile(AdamW* opt, const TrainConfig& cfg) {
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
    opt->set_schedule(cfg.schedule, cfg.warmup_steps, cfg.train_steps);
    opt->set_weight_decay(cfg.weight_decay);
    loss_scale_ = cfg.mixed_precision ? cfg.loss_scale : 1.0f;
    loss_scale_interval_ = cfg.loss_scale_interval;
    if (cfg.mixed_precision) init_mixed_precision();
}

void Trainer::init_mixed_precision() {
    // Initialize mixed precision training state
    loss_scale_ = loss_scale_ > 0.0f ? loss_scale_ : 1024.0f;
    steps_since_scale_update_ = 0;
}

float Trainer::eval_loss(DataLoader& val_dl, int64_t max_batches) {
    Tensor input_ids(Shape{val_dl.batch_size(), val_dl.seq_length()}, DType::F32);
    Tensor labels(Shape{val_dl.batch_size(), val_dl.seq_length()}, DType::F32);
    float total_loss = 0;
    int64_t count = 0;
    AutogradEngine::set_enabled(false);
    val_dl.reset();
    while (val_dl.next_batch(input_ids, labels) && count < max_batches) {
        int64_t B = input_ids.dim(0);
        int64_t S = input_ids.dim(1);
        Tensor positions(Shape{B, S}, DType::F32);
        float* pd = positions.data<float>();
        for (int64_t i = 0; i < B * S; i++)
            pd[i] = (float)(i % S);
        Tensor logits = model_->forward(input_ids, positions);
        Tensor loss = AutogradEngine::cross_entropy_op(logits, labels);
        total_loss += *(const float*)loss.data();
        count++;
    }
    AutogradEngine::set_enabled(true);
    return count > 0 ? total_loss / (float)count : 0;
}

void Trainer::unscale_gradients(float scale) {
    if (scale == 1.0f) return;
    for (auto* p : model_params_) {
        if (!p->has_grad()) continue;
        float* g = p->grad().data<float>();
        int64_t n = p->grad().numel();
        for (int64_t i = 0; i < n; i++)
            g[i] /= scale;
    }
}

void Trainer::fit(DataLoader& dl, const TrainConfig& cfg,
                  DataLoader* val_dl) {
    Tensor input_ids(Shape{cfg.batch_size, cfg.seq_length}, DType::F32);
    Tensor labels(Shape{cfg.batch_size, cfg.seq_length}, DType::F32);
    int acc_steps = cfg.gradient_accumulation_steps > 0 ? cfg.gradient_accumulation_steps : 1;

    loss_scale_ = cfg.mixed_precision ? cfg.loss_scale : 1.0f;
    steps_since_scale_update_ = 0;

    // C9: Curriculum setup
    if (cfg.curriculum) dl.set_curriculum(cfg.curriculum_epochs);

    auto epoch_start = std::chrono::steady_clock::now();
    int64_t tokens_processed = 0;

    for (int epoch = 0; epoch < cfg.num_epochs; epoch++) {
        dl.reset();
        dl.shuffle(epoch);
        // C9: Curriculum step
        if (cfg.curriculum) dl.curriculum_step(epoch);

        while (dl.next_batch(input_ids, labels)) {
            // C8: Data augmentation
            if (cfg.data_augmentation) dl.apply_augmentation(input_ids, labels);

            float loss = 0;
            optimizer_->zero_grad();
            float scale = cfg.mixed_precision ? loss_scale_ : 1.0f;
            for (int acc = 0; acc < acc_steps; acc++) {
                loss = micro_step(input_ids, labels, scale);
            }

            // C19: R-Drop consistency
            if (cfg.use_rdrop && cfg.rdrop_alpha > 0) {
                loss += rdrop_loss(input_ids, labels, cfg.rdrop_alpha);
            }

            if (cfg.mixed_precision) unscale_gradients(loss_scale_);
            float grad_norm = clip_gradients(cfg.max_grad_norm);
            if (cfg.mixed_precision) dynamic_loss_scale(grad_norm, cfg.max_grad_norm);
            optimizer_->step();

            // C20: EMA tracking
            if (ema_enabled_) ema_step();

            if (!codebooks_.empty() && step_ % 100 == 0) {
                for (auto* cb : codebooks_) cb->ema_update(0.999f);
            }

            tokens_processed += cfg.batch_size * cfg.seq_length * acc_steps;
            auto now = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - epoch_start).count();
            int tokens_per_sec = elapsed_ms > 0 ? (int)(tokens_processed * 1000 / (std::max)(elapsed_ms, (int64_t)1)) : 0;

            metrics_.loss = loss;
            metrics_.perplexity = std::exp(loss);
            metrics_.grad_norm = grad_norm;
            metrics_.learning_rate = optimizer_ ? optimizer_->get_lr() : cfg.learning_rate;
            metrics_.tokens_per_sec = tokens_per_sec;
            metrics_.step = step_;
            metrics_.epoch = epoch;

            if (step_ % cfg.log_interval == 0) {
                if (log_cb_) log_cb_(metrics_);
                if (step_cb_) step_cb_(step_, metrics_);
            }
            if (step_ % cfg.val_interval == 0 && val_dl) {
                float val_loss = eval_loss(*val_dl, 20);
                metrics_.val_loss = val_loss;
                metrics_.val_perplexity = std::exp(val_loss);
            }
            if (step_ % cfg.save_interval == 0 && cfg.save_interval > 0) {
                save_checkpoint(cfg.output_path);
            }
            step_++;
        }
        if (epoch_cb_) epoch_cb_(epoch, metrics_);
    }
    // C20: Apply EMA at end of training if enabled
    if (ema_enabled_) ema_apply();
    save_checkpoint(cfg.output_path);
}

float Trainer::micro_step(const Tensor& input_ids, const Tensor& labels, float loss_scale) {
    if (!optimizer_) return 0;
    int64_t B = input_ids.dim(0);
    int64_t S = input_ids.dim(1);

    Tensor positions(Shape{B, S}, DType::F32);
    float* pd = positions.data<float>();
    for (int64_t i = 0; i < B * S; i++)
        pd[i] = (float)(i % S);

    // Mixed precision: convert inputs to FP16 and back to simulate FP16 forward precision loss
    Tensor fp_input = input_ids;
    Tensor fp_labels = labels;
    if (loss_scale != 1.0f) {
        fp_input = input_ids.to_dtype(DType::F16).to_dtype(DType::F32);
        fp_labels = labels.to_dtype(DType::F16).to_dtype(DType::F32);
    }

    AutogradEngine::set_enabled(true);
    auto& engine = AutogradEngine::instance();
    for (auto* p : model_params_) {
        engine.register_parameter(p);
    }
    Tensor logits = model_->forward(fp_input, positions);
    Tensor loss = AutogradEngine::cross_entropy_op(logits, fp_labels);
    if (loss_scale != 1.0f) {
        float* ld = (float*)loss.data();
        *ld *= loss_scale;
    }
    engine.backward(loss);
    engine.clear();
    AutogradEngine::set_enabled(false);
    return *(const float*)loss.data() / loss_scale;
}

float Trainer::train_step(const Tensor& input_ids, const Tensor& labels) {
    if (!optimizer_) return 0;
    optimizer_->zero_grad();
    // Use current loss_scale_ (cfg.mixed_precision determines if != 1.0f)
    float loss = micro_step(input_ids, labels, loss_scale_);
    if (loss_scale_ != 1.0f) {
        unscale_gradients(loss_scale_);
    }
    float grad_norm = clip_gradients(1.0f);
    if (loss_scale_ != 1.0f) {
        if (grad_norm > 1.0f) {
            loss_scale_ = std::max(1.0f, loss_scale_ / 2.0f);
            steps_since_scale_update_ = 0;
        } else {
            steps_since_scale_update_++;
            if (steps_since_scale_update_ >= loss_scale_interval_) {
                loss_scale_ = std::min(65536.0f, loss_scale_ * 2.0f);
                steps_since_scale_update_ = 0;
            }
        }
    }
    optimizer_->step();
    return loss;
}

float Trainer::clip_gradients(float max_norm) {
    if (max_norm <= 0 || model_params_.empty()) return 0;
    float total_norm = 0;
    for (auto* p : model_params_) {
        if (!p->has_grad()) continue;
        const float* g = p->grad().data<float>();
        int64_t n = p->grad().numel();
        float sq_sum = 0;
        for (int64_t i = 0; i < n; i++) sq_sum += g[i] * g[i];
        total_norm += sq_sum;
    }
    total_norm = std::sqrt(total_norm);
    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-8f);
        for (auto* p : model_params_) {
            if (!p->has_grad()) continue;
            float* g = p->grad().data<float>();
            int64_t n = p->grad().numel();
            for (int64_t i = 0; i < n; i++) g[i] *= scale;
        }
    }
    return total_norm;
}

void Trainer::save_checkpoint(const std::string& path) {
    model_->save(path);
    if (!optimizer_) return;
    std::string opt_path = path + ".opt";
    FILE* fp = std::fopen(opt_path.c_str(), "wb");
    if (!fp) return;
    int32_t step_i = (int32_t)step_;
    float lr = optimizer_->get_lr();
    fwrite(&step_i, sizeof(step_i), 1, fp);
    fwrite(&lr, sizeof(lr), 1, fp);
    int32_t num_params = (int32_t)model_params_.size();
    fwrite(&num_params, sizeof(num_params), 1, fp);
    for (auto* p : model_params_) {
        auto& state = optimizer_->get_state(p);
        int64_t n = p->numel();
        int64_t written_m = 0, written_v = 0;
        if (state.m.numel() > 0) {
            written_m = state.m.numel();
            fwrite(&written_m, sizeof(written_m), 1, fp);
            fwrite(state.m.data<float>(), (size_t)written_m * sizeof(float), 1, fp);
        } else {
            fwrite(&written_m, sizeof(written_m), 1, fp);
        }
        if (state.v.numel() > 0) {
            written_v = state.v.numel();
            fwrite(&written_v, sizeof(written_v), 1, fp);
            fwrite(state.v.data<float>(), (size_t)written_v * sizeof(float), 1, fp);
        } else {
            fwrite(&written_v, sizeof(written_v), 1, fp);
        }
    }
    fclose(fp);
}

void Trainer::load_checkpoint(const std::string& path) {
    model_->load(path);
    if (!optimizer_) return;
    std::string opt_path = path + ".opt";
    FILE* fp = std::fopen(opt_path.c_str(), "rb");
    if (!fp) return;
    int32_t step_i = 0;
    float lr = 0;
    fread(&step_i, sizeof(step_i), 1, fp);
    fread(&lr, sizeof(lr), 1, fp);
    step_ = (int)step_i;
    optimizer_->set_lr(lr);
    int32_t num_params = 0;
    fread(&num_params, sizeof(num_params), 1, fp);
    for (int32_t i = 0; i < num_params && i < (int32_t)model_params_.size(); i++) {
        auto* p = model_params_[(size_t)i];
        auto& state = optimizer_->get_state(p);
        int64_t n = p->numel();
        int64_t read_m = 0, read_v = 0;
        fread(&read_m, sizeof(read_m), 1, fp);
        if (read_m > 0 && read_m <= n) {
            state.m = Tensor::zeros(p->shape());
            fread(state.m.data<float>(), (size_t)read_m * sizeof(float), 1, fp);
        }
        fread(&read_v, sizeof(read_v), 1, fp);
        if (read_v > 0 && read_v <= n) {
            state.v = Tensor::zeros(p->shape());
            fread(state.v.data<float>(), (size_t)read_v * sizeof(float), 1, fp);
        }
    }
    fclose(fp);
}

void Trainer::set_log_callback(LogCallback cb) {
    log_cb_ = cb;
}

void Trainer::set_epoch_callback(EpochCallback cb) {
    epoch_cb_ = cb;
}

void Trainer::set_step_callback(StepCallback cb) {
    step_cb_ = cb;
}

const TrainMetrics& Trainer::metrics() const {
    return metrics_;
}

// Helper: create position tensor from input shape
static Tensor positions_from(const Tensor& input_ids) {
    int64_t B = input_ids.dim(0);
    int64_t S = input_ids.dim(1);
    Tensor pos({B, S});
    float* pd = pos.data<float>();
    for (int64_t i = 0; i < B * S; i++)
        pd[i] = (float)(i % S);
    return pos;
}

// C19: R-Drop consistency — forward twice, compute KL between output distributions
float Trainer::rdrop_loss(const Tensor& input_ids, const Tensor& labels, float alpha) {
    Tensor logits1 = model_->forward(input_ids, positions_from(input_ids));
    Tensor logits2 = model_->forward(input_ids, positions_from(input_ids));
    int64_t B = logits1.dim(0), S = logits1.dim(1), V = logits1.dim(2);
    float kl = 0;
    const float* l1 = logits1.data<float>();
    const float* l2 = logits2.data<float>();
    for (int64_t i = 0; i < B * S; i++) {
        const float* row1 = l1 + i * V;
        const float* row2 = l2 + i * V;
        float max1 = -INFINITY, max2 = -INFINITY;
        for (int64_t v = 0; v < V; v++) {
            if (row1[v] > max1) max1 = row1[v];
            if (row2[v] > max2) max2 = row2[v];
        }
        float sum1 = 0, sum2 = 0;
        std::vector<float> p1(V), p2(V);
        for (int64_t v = 0; v < V; v++) {
            p1[v] = std::exp(row1[v] - max1); sum1 += p1[v];
            p2[v] = std::exp(row2[v] - max2); sum2 += p2[v];
        }
        float inv1 = 1.0f / (sum1 + 1e-10f);
        float inv2 = 1.0f / (sum2 + 1e-10f);
        for (int64_t v = 0; v < V; v++) {
            float prob1 = p1[v] * inv1;
            float prob2 = p2[v] * inv2;
            kl += prob1 * (std::log(prob1 + 1e-10f) - std::log(prob2 + 1e-10f));
        }
    }
    float ce_loss = *(const float*)AutogradEngine::cross_entropy_op(logits1, labels).data();
    return ce_loss + alpha * kl / (float)(B * S);
}

// C17: Label smoothing cross-entropy
Tensor Trainer::label_smoothing_loss(const Tensor& logits, const Tensor& labels, float smoothing) {
    int64_t B = logits.dim(0);
    int64_t S = logits.dim(1);
    int64_t V = logits.dim(2);
    Tensor loss({B * S});
    const float* ld = labels.data<float>();
    const float* lp = logits.data<float>();
    float* ld_ = loss.data<float>();
    for (int64_t i = 0; i < B * S; i++) {
        int label = (int)ld[i];
        if (label < 0 || label >= V) { ld_[i] = 0; continue; }
        const float* logit_row = lp + i * V;
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++)
            if (logit_row[v] > max_l) max_l = logit_row[v];
        float sum_exp = 0;
        std::vector<float> p(V);
        for (int64_t v = 0; v < V; v++) {
            p[v] = std::exp(logit_row[v] - max_l);
            sum_exp += p[v];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-10f);
        float smooth_loss = 0;
        float uniform = smoothing / (float)V;
        for (int64_t v = 0; v < V; v++) {
            float prob = p[v] * inv_sum;
            float target = (v == label) ? (1.0f - smoothing) : 0.0f;
            target += uniform;
            smooth_loss -= target * std::log(prob + 1e-10f);
        }
        ld_[i] = smooth_loss;
    }
    return loss.reshape({B, S});
}

// C20: EMA tracking
void Trainer::ema_init(float decay) {
    ema_decay_ = decay;
    ema_enabled_ = true;
    ema_params_.clear();
    for (auto* p : model_params_) {
        ema_params_.push_back(Tensor::zeros(p->shape()));
    }
}

void Trainer::ema_step() {
    if (!ema_enabled_) return;
    for (size_t i = 0; i < model_params_.size(); i++) {
        float* e = ema_params_[i].data<float>();
        const float* p = model_params_[i]->data<float>();
        int64_t n = model_params_[i]->numel();
        for (int64_t j = 0; j < n; j++)
            e[j] = ema_decay_ * e[j] + (1.0f - ema_decay_) * p[j];
    }
}

void Trainer::ema_apply() {
    if (!ema_enabled_) return;
    for (size_t i = 0; i < model_params_.size(); i++) {
        std::memcpy(model_params_[i]->data<float>(),
                    ema_params_[i].data<float>(),
                    model_params_[i]->numel() * sizeof(float));
    }
}

void Trainer::ema_swap() {
    if (!ema_enabled_) return;
    for (size_t i = 0; i < model_params_.size(); i++) {
        float* p = model_params_[i]->data<float>();
        float* e = ema_params_[i].data<float>();
        int64_t n = model_params_[i]->numel();
        for (int64_t j = 0; j < n; j++) {
            float tmp = p[j];
            p[j] = e[j];
            e[j] = tmp;
        }
    }
}

// C14: Enhanced mixed precision
void Trainer::dynamic_loss_scale(float grad_norm, float max_grad_norm) {
    if (loss_scale_ == 1.0f) return;
    if (grad_norm > max_grad_norm) {
        loss_scale_ = std::max(1.0f, loss_scale_ / 2.0f);
        steps_since_scale_update_ = 0;
    } else {
        steps_since_scale_update_++;
        if (steps_since_scale_update_ >= loss_scale_interval_) {
            loss_scale_ = std::min(65536.0f, loss_scale_ * 2.0f);
            steps_since_scale_update_ = 0;
        }
    }
}

} // namespace oil
