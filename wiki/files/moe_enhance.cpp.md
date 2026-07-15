# `moe_enhance.cpp` — MoE Enhancement Implementation

**Path:** `src/moe_enhance.cpp`

Enhanced MoE features: load balancing, auxiliary losses, and routing improvements.

## Implementations

| Feature | Description |
|---------|-------------|
| Load Balancing Loss | Penalizes uneven expert utilization |
| Z-Loss | `z_loss = 1e-4 * log(∑exp(logits))²` stabilizes router |
| Router Z-Loss | Auxiliary loss on router logits |
| Expert Dropout | Dropout within expert computation |
| Batch Prioritized Routing | Ensures tokens routed to experts with capacity |

## Loss Terms

```cpp
// Total loss = main_loss + λ₁ * balancing_loss + λ₂ * z_loss
float balancing_loss = compute_load_balancing_loss(routing_weights, expert_counts);
float z_loss = compute_z_loss(router_logits);
float total_loss = main_loss + 0.01 * balancing_loss + 0.0001 * z_loss;
```

## Load Balancing

The load balancing loss encourages uniform expert utilization:
```
balancing_loss = Σ(importance_normalized * frequency_normalized)
```
