# `production.cpp` — Production Deployment

**Path:** `src/production.cpp`

Production deployment features: optimized serving, batching, and monitoring.

## Functions

| Function | Description |
|----------|-------------|
| `setup_production_env()` | Configure for production |
| `serve_model(model, port)` | Start HTTP inference server |
| `batch_infer(requests)` | Batch multiple inference requests |
| `warm_up()` | Pre-warm model on GPU |

## Production Config

```cpp
InferenceConfig{};
// For production:
config.max_batch_size = 32;     // Larger batches
config.num_threads = 8;          // More threads
config.use_flash_attn = true;    // Memory efficient
config.quantize_matmul = true;   // Lower latency
```
