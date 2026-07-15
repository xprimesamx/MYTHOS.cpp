# `moe.h` — Mixture-of-Experts Core

**Path:** `include/oil/moe.h`

Core MoE (Mixture-of-Experts) implementation with routing and expert computation.

## MoE Classes

```cpp
class MoE {
    int num_experts, top_k, expert_dim;
    Tensor router_weights;   // router network
    std::vector<FeedForward> experts;
    
    Tensor forward(const Tensor& x);
    Tensor router(const Tensor& x);
    Tensor dispatch(const Tensor& x, const Tensor& routing_weights, 
                    const Tensor& expert_indices);
};
```

### MoEConfig

```cpp
struct MoEConfig {
    int num_experts = 8;
    int top_k = 2;
    int expert_hidden_size = 2048;
    float router_jitter_noise = 0.01f;
    bool z_loss = true;
};
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Top-K Routing** | Each token activates only top-k experts |
| **Load Balancing** | Auxiliary loss to balance expert utilization |
| **Router Jitter** | Noise during training to improve routing diversity |
| **Z-Loss** | Stabilizes router training |

### MoE Variants

| File | Description |
|------|-------------|
| `moe.h/cpp` | Core MoE implementation |
| `moe_enhance.h/cpp` | Enhanced MoE features |
| `moe_variants.h/cpp` | MoE architecture variants |

### Training MoEs

MoE training uses:
- Sparse expert computation (only activated experts run)
- Load balancing loss to prevent expert collapse
- Sequence-level auxiliary losses
- Gradient computation only for selected experts
