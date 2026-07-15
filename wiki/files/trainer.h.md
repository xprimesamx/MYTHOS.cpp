# `trainer.h` — Training Engine

**Path:** `include/oil/trainer.h`

The training engine for training models from scratch with AdamW optimizer and data loading.

## Classes

### AdamW Optimizer
```cpp
class AdamW {
    float lr, beta1, beta2, eps, weight_decay;
    int step;
    
    void step(std::vector<Tensor*>& params);
    void zero_grad();
};
```
AdamW optimizer with:
- Default: lr=3e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1
- Cosine learning rate scheduling
- Gradient clipping support

### TrainerConfig
```cpp
struct TrainerConfig {
    int64_t batch_size = 8;
    int64_t epochs = 10;
    int64_t warmup_steps = 100;
    int64_t max_steps = 0;         // 0 = no limit
    float learning_rate = 3e-4f;
    float weight_decay = 0.1f;
    float grad_clip = 1.0f;
    bool use_moe = false;
    // ... more configuration params
};
```

### Trainer
```cpp
class Trainer {
    std::unique_ptr<Model> model;
    std::unique_ptr<AdamW> optimizer;
    std::unique_ptr<DataLoader> dataloader;
    
    void train();
    void train_step(const Tensor& batch);
    void save_checkpoint(const std::string& path);
    float get_loss() const;
};
```
The main training loop handler.

### DataLoader
```cpp
class DataLoader {
    DataLoader(const std::string& data_path, int64_t batch_size, int64_t seq_len);
    bool next_batch(Tensor& input_ids, Tensor& target_ids);
    int64_t num_batches() const;
    void shuffle();
};
```
Text data loader:
- Reads raw text files
- Tokenizes on the fly
- Creates batched sequences with input/target pairs
- Supports shuffling
