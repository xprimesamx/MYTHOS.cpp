# `engines/trainer/moe/moe.cpp` — MoE Trainer

**Path:** `engines/trainer/moe/moe.cpp`

Mixture-of-Experts training engine — trains sparse MoE models with expert routing.

## MoETrainer

```cpp
class MoETrainer {
    int num_experts;
    int top_k;
    float load_balancing_coef;
    float z_loss_coef;
    
    void train_step(const Tensor& batch);
    void compute_router_loss(const Tensor& routing_weights);
    void update_experts(const Tensor& gradients, const Tensor& expert_mask);
    void save_checkpoint(const std::string& path);
};
```

### Training Process

1. **Forward**: Each token → router → top-k experts → weighted sum
2. **Loss**: Main loss + load balancing loss + z-loss
3. **Backward**: Gradients flow only through selected experts
4. **Update**: Experts updated sparsely based on routing

### MoE Multimodal Modules

| Module | File | Description |
|--------|------|-------------|
| Audio | `audio/audio.cpp` | Audio processing expert |
| Vision | `vision/vision.cpp` | Image processing expert |
| Video | `video/video.cpp` | Video processing expert |
| Text | `text/multimodal_text.cpp` | Multimodal text expert |
| OCR | `ocr/ocr.cpp` | OCR processing expert |
| Embeddings | `embeddings/embeddings.cpp` | Embedding model expert |

```cpp
// Example: MoE multimodal training
MoETrainer trainer(num_experts=8, top_k=2);
// Expert 0-3: text domains
// Expert 4-5: vision domains  
// Expert 6: audio domain
// Expert 7: video domain
```
