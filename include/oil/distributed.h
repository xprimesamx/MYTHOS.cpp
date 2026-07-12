#pragma once
#include "oil/tensor.h"
#include "oil/types.h"
#include "oil/model.h"
#include "oil/transformer.h"
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>

namespace oil {

// C23: Distributed Data Parallel (DDP) context
// Uses shared-memory thread-based parallelism (simulates multi-GPU)
class DistributedContext {
public:
    enum class Mode { DDP, TENSOR_PARALLEL, PIPELINE_PARALLEL };

    DistributedContext(int world_size, int world_rank, Mode mode = Mode::DDP);
    ~DistributedContext();

    int world_size() const { return world_size_; }
    int world_rank() const { return world_rank_; }
    Mode mode() const { return mode_; }
    bool is_main() const { return world_rank_ == 0; }

    // All-reduce across all ranks (sum)
    void all_reduce(Tensor& tensor);
    void all_reduce(float* data, int64_t n);
    void barrier();

    // Broadcast from main rank
    void broadcast(Tensor& tensor, int src_rank = 0);
    void broadcast(float* data, int64_t n, int src_rank = 0);

    // All-gather
    void all_gather(const Tensor& local, Tensor& global);

private:
    int world_size_;
    int world_rank_;
    Mode mode_;
    std::mutex barrier_mutex_;
    std::condition_variable barrier_cv_;
    int barrier_count_ = 0;
};

// C23: DDP wrapper — replicates model across ranks, synchronizes gradients
class DDPWrapper {
public:
    DDPWrapper(Model* model, int world_size, int world_rank);
    void sync_gradients();
    DistributedContext& context() { return ctx_; }
    Model* model() { return local_model_; }
private:
    DistributedContext ctx_;
    Model* local_model_;
    std::vector<Tensor*> params_;
};

// C24: Tensor Parallel — splits linear layers across ranks
class TensorParallelWrapper {
public:
    TensorParallelWrapper(Model* model, int world_size, int world_rank);
    Tensor forward(const Tensor& input, const Tensor& positions);
    DistributedContext& context() { return ctx_; }
private:
    DistributedContext ctx_;
    Model* model_;
    void split_linear(const Tensor& weight, Tensor& local_weight,
                      int64_t dim, bool is_column);
};

// C25: Pipeline Parallel — splits layers across ranks
class PipelineParallelWrapper {
public:
    PipelineParallelWrapper(Model* model, int world_size, int world_rank);
    Tensor forward(const Tensor& input, const Tensor& positions,
                   const Tensor& mask, KVCache& cache);
    DistributedContext& context() { return ctx_; }
private:
    DistributedContext ctx_;
    Model* model_;
    int64_t micro_batches_ = 4;
};

} // namespace oil
