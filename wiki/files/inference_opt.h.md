# `inference_opt.h` — Optimized Inference

**Path:** `include/oil/inference_opt.h`

Optimized inference configurations and utilities for production deployment.

## InferenceConfig

```cpp
struct InferenceConfig {
    int max_batch_size = 1;
    int max_seq_len = 2048;
    bool use_flash_attn = true;
    bool use_kv_cache = true;
    bool quantize_matmul = true;
    int num_threads = 4;
};
```

## Optimized Engine

```cpp
class OptimizedInference {
    std::unique_ptr<Model> model;
    InferenceConfig config;
    
    OptimizedInference(std::unique_ptr<Model> model, InferenceConfig cfg);
    
    Tensor forward_optimized(const Tensor& input, KVCache* cache);
    void prefill(const Tensor& input);
    Tensor decode_step(const Tensor& token);
    void set_num_threads(int n);
};
```

### Optimizations

| Optimization | Description |
|-------------|-------------|
| Flash Attention | Memory-efficient attention |
| KV Cache | Cached K,V for generation |
| Quantized Matmul | Matrix multiply with quantized weights |
| Thread Pool | Parallel computation |
| Batch Prefill | Efficient prompt processing |

### Memory Usage

| Config (seq_len) | FP16 | OIL4 | OIL8 |
|-----------------|------|------|------|
| 2048 | ~1GB | ~256MB | ~128MB |
| 8192 | ~4GB | ~1GB | ~512MB |
