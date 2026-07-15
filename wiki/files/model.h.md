# `model.h` — Model Interface

**Path:** `include/oil/model.h`

Defines the abstract `Model` interface and the `DenseModel` implementation.

## Model (Abstract Base)

```cpp
class Model {
public:
    TransformerConfig config;
    
    virtual Tensor forward(const Tensor& input_ids, 
                           const Tensor& positions,
                           KVCache* cache = nullptr) = 0;
    virtual void load(const std::string& oil_path);
    virtual void save(const std::string& oil_path) const;
    virtual int64_t param_count() const = 0;
    virtual int64_t vocab_size() const = 0;
};
```

| Method | Description |
|--------|-------------|
| `forward()` | Run forward pass through the model |
| `load()` | Load model weights from `.oil` file |
| `save()` | Save model weights to `.oil` file |
| `param_count()` | Get total parameter count |
| `vocab_size()` | Get vocabulary size |

## DenseModel

```cpp
class DenseModel : public Model {
    std::unique_ptr<Embedding> tok_embeddings;
    std::vector<std::unique_ptr<TransformerBlock>> layers;
    std::unique_ptr<RMSNorm> norm;
    std::unique_ptr<Linear> lm_head;
};
```

Standard dense transformer implementation with:
- Token embeddings
- Stack of transformer blocks
- Final RMS normalization
- Language model head (linear projection to vocab)
