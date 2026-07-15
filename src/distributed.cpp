#include "oil/distributed.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <condition_variable>

namespace oil {

// ===========================================================================
// DistributedContext — shared-memory all-reduce, broadcast, barrier
// ===========================================================================

static std::mutex global_reduce_mutex;
static std::vector<float> global_reduce_buffer;

DistributedContext::DistributedContext(int world_size, int world_rank, Mode mode)
    : world_size_(world_size), world_rank_(world_rank), mode_(mode) {}

DistributedContext::~DistributedContext() {}

void DistributedContext::barrier() {
    std::unique_lock<std::mutex> lock(barrier_mutex_);
    barrier_count_++;
    if (barrier_count_ < world_size_) {
        barrier_cv_.wait(lock, [this] { return barrier_count_ >= world_size_; });
    } else {
        barrier_count_ = 0;
        barrier_cv_.notify_all();
    }
}

void DistributedContext::all_reduce(float* data, int64_t n) {
    if (world_size_ <= 1) return;
    {
        std::lock_guard<std::mutex> lock(global_reduce_mutex);
        if (global_reduce_buffer.size() < (size_t)n)
            global_reduce_buffer.resize(n, 0.0f);
        for (int64_t i = 0; i < n; i++)
            global_reduce_buffer[i] += data[i];
    }
    barrier();
    {
        std::lock_guard<std::mutex> lock(global_reduce_mutex);
        std::memcpy(data, global_reduce_buffer.data(), n * sizeof(float));
        std::fill(global_reduce_buffer.begin(), global_reduce_buffer.begin() + n, 0.0f);
    }
    barrier();
}

void DistributedContext::all_reduce(Tensor& tensor) {
    all_reduce((float*)tensor.data(), tensor.numel());
}

void DistributedContext::broadcast(float* data, int64_t n, int src_rank) {
    if (world_size_ <= 1) return;
    if (world_rank_ == src_rank) {
        std::lock_guard<std::mutex> lock(global_reduce_mutex);
        if (global_reduce_buffer.size() < (size_t)n)
            global_reduce_buffer.resize(n);
        std::memcpy(global_reduce_buffer.data(), data, n * sizeof(float));
    }
    barrier();
    {
        std::lock_guard<std::mutex> lock(global_reduce_mutex);
        if (world_rank_ != src_rank)
            std::memcpy(data, global_reduce_buffer.data(), n * sizeof(float));
    }
    barrier();
}

void DistributedContext::broadcast(Tensor& tensor, int src_rank) {
    broadcast((float*)tensor.data(), tensor.numel(), src_rank);
}

void DistributedContext::all_gather(const Tensor& local, Tensor& global) {
    if (world_size_ <= 1) { global.copy_from(local); return; }
    int64_t local_n = local.numel();
    int64_t total_n = local_n * world_size_;
    if (global.numel() != total_n)
        global = Tensor({total_n});
    {
        std::lock_guard<std::mutex> lock(global_reduce_mutex);
        if (global_reduce_buffer.size() < (size_t)total_n)
            global_reduce_buffer.resize(total_n);
        std::memcpy(&global_reduce_buffer[world_rank_ * local_n],
                    local.data<float>(), local_n * sizeof(float));
    }
    barrier();
    {
        std::lock_guard<std::mutex> lock(global_reduce_mutex);
        std::memcpy(global.data<float>(), global_reduce_buffer.data(),
                    total_n * sizeof(float));
    }
    barrier();
}

// ===========================================================================
// DDPWrapper — gradient synchronization across ranks
// ===========================================================================

DDPWrapper::DDPWrapper(Model* model, int world_size, int world_rank)
    : ctx_(world_size, world_rank, DistributedContext::Mode::DDP),
      local_model_(model) {
    // Collect all trainable parameters
    DenseModel* dm = dynamic_cast<DenseModel*>(model);
    if (dm) {
        params_.push_back(&dm->tok_embeddings->weight);
        for (auto& layer : dm->layers) {
            params_.push_back(&layer->attention_norm.weight);
            params_.push_back(&layer->attention.q_proj.weight);
            params_.push_back(&layer->attention.q_proj.bias);
            params_.push_back(&layer->attention.k_proj.weight);
            params_.push_back(&layer->attention.k_proj.bias);
            params_.push_back(&layer->attention.v_proj.weight);
            params_.push_back(&layer->attention.v_proj.bias);
            params_.push_back(&layer->attention.o_proj.weight);
            params_.push_back(&layer->attention.o_proj.bias);
            params_.push_back(&layer->ffn_norm.weight);
            params_.push_back(&layer->ffn.gate_proj.weight);
            params_.push_back(&layer->ffn.gate_proj.bias);
            params_.push_back(&layer->ffn.up_proj.weight);
            params_.push_back(&layer->ffn.up_proj.bias);
            params_.push_back(&layer->ffn.down_proj.weight);
            params_.push_back(&layer->ffn.down_proj.bias);
        }
        params_.push_back(&dm->norm->weight);
        params_.push_back(&dm->lm_head->weight);
        params_.push_back(&dm->lm_head->bias);
    }
}

void DDPWrapper::sync_gradients() {
    for (auto* p : params_) {
        if (p->has_grad())
            ctx_.all_reduce(p->grad());
    }
}

// ===========================================================================
// TensorParallelWrapper — split linear layers across ranks
// ===========================================================================

TensorParallelWrapper::TensorParallelWrapper(Model* model, int world_size, int world_rank)
    : ctx_(world_size, world_rank, DistributedContext::Mode::TENSOR_PARALLEL),
      model_(model) {}

static void column_split(const Tensor& weight, Tensor& local, int rank, int size) {
    int64_t out_dim = weight.dim(0);
    int64_t in_dim = weight.dim(1);
    int64_t chunk = (out_dim + size - 1) / size;
    int64_t start = rank * chunk;
    int64_t end = std::min(start + chunk, out_dim);
    local = weight.slice(0, start, end);
}

static void row_split(const Tensor& weight, Tensor& local, int rank, int size) {
    int64_t out_dim = weight.dim(0);
    int64_t in_dim = weight.dim(1);
    int64_t chunk = (in_dim + size - 1) / size;
    int64_t start = rank * chunk;
    int64_t end = std::min(start + chunk, in_dim);
    local = weight.slice(1, start, end);
}

void TensorParallelWrapper::split_linear(const Tensor& weight, Tensor& local_weight,
                                           int64_t dim, bool is_column) {
    if (is_column)
        column_split(weight, local_weight, ctx_.world_rank(), ctx_.world_size());
    else
        row_split(weight, local_weight, ctx_.world_rank(), ctx_.world_size());
}

Tensor TensorParallelWrapper::forward(const Tensor& input, const Tensor& positions) {
    return model_->forward(input, positions);
}

// ===========================================================================
// PipelineParallelWrapper — split layers across ranks
// ===========================================================================

PipelineParallelWrapper::PipelineParallelWrapper(Model* model, int world_size, int world_rank)
    : ctx_(world_size, world_rank, DistributedContext::Mode::PIPELINE_PARALLEL) {
    (void)model;
}

Tensor PipelineParallelWrapper::forward(const Tensor& input,
                                         const Tensor& positions,
                                         const Tensor& mask,
                                         KVCache& cache) {
    // Pipeline parallel forwards the full model on this micro-batch
    return model_->forward(input, positions, &cache);
}

} // namespace oil
