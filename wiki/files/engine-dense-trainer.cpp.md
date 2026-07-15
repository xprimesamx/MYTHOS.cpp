# `engines/trainer/dense/trainer.cpp` — Dense Trainer

**Path:** `engines/trainer/dense/trainer.cpp`

Standard dense transformer training implementation.

## DenseTrainer

```cpp
class DenseTrainer {
    std::unique_ptr<Model> model;
    std::unique_ptr<AdamW> optimizer;
    std::unique_ptr<DataLoader> dataloader;
    
    DenseTrainer(const TransformerConfig& model_cfg, 
                 const TrainerConfig& train_cfg);
    
    void train(int epochs);
    float train_epoch();
    float train_batch(const Tensor& input, const Tensor& target);
    void save(const std::string& path);
    void load(const std::string& path);
};
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `trainer.cpp` | Dense trainer | Training loop, backprop |
| `dataloader.cpp` | Data loader | Batch management |
| `checkpoint.cpp` | Checkpointing | Save/resume training |

### Dataloader

```cpp
class DataLoader {
    DataLoader(const std::string& data_path, int batch_size, int seq_len);
    bool next(Tensor& input, Tensor& target);
    void reset();
    int num_batches() const;
};
```

### Checkpoint

```cpp
class Checkpoint {
    void save(const std::string& path, const Model& model, 
              const Optimizer& opt, int step);
    void load(const std::string& path, Model& model, 
              Optimizer& opt, int& step);
};
```
