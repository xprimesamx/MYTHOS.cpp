# `optimizer.h` — Training Optimizers

**Path:** `include/oil/optimizer.h`

Optimizer implementations for training neural networks.

## Available Optimizers

### AdamW (default)
```cpp
class AdamW {
    float lr, beta1, beta2, eps, weight_decay;
    std::vector<Tensor> m;  // first moment
    std::vector<Tensor> v;  // second moment
    
    AdamW(float lr = 3e-4f, float beta1 = 0.9f, 
          float beta2 = 0.95f, float eps = 1e-8f,
          float weight_decay = 0.1f);
    
    void step(const std::vector<Tensor*>& params);
    void zero_grad();
    void lr_schedule_cosine(int step, int warmup_steps, int total_steps);
};
```

### SGD
```cpp
class SGD {
    float lr, momentum, weight_decay;
    
    SGD(float lr = 1e-3f, float momentum = 0.9f, float weight_decay = 0.0f);
    void step(const std::vector<Tensor*>& params);
    void zero_grad();
};
```

### Features

| Feature | AdamW | SGD |
|---------|-------|-----|
| Momentum | ✓ (Adam) | ✓ |
| Weight decay | ✓ (decoupled) | ✓ |
| Cosine LR schedule | ✓ | ✓ |
| Gradient clipping | ✓ | ✓ |
| Warmup | ✓ | ✓ |
