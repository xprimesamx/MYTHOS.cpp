# `optimizer.cpp` — Optimizer Implementation

**Path:** `src/optimizer.cpp`

AdamW optimizer implementation with weight decay correction.

## AdamW Algorithm

```
for each step t:
    g = ∇L(θ)               // gradients
    m = β1·m + (1-β1)·g      // first moment estimate
    v = β2·v + (1-β2)·g²     // second moment estimate
    m̂ = m / (1 - β1ᵗ)         // bias correction
    v̂ = v / (1 - β2ᵗ)
    θ = θ - lr · (m̂ / (√v̂ + ε) + λ·θ)  // update with decoupled weight decay
```

## Functions

| Function | Description |
|----------|-------------|
| `AdamW::step()` | Update all parameters |
| `AdamW::zero_grad()` | Reset gradients to zero |
| `AdamW::lr_schedule_cosine()` | Cosine LR decay with warmup |

## SGD Algorithm

```
v = μ·v + g               // momentum
θ = θ - lr · (v + λ·θ)    // update with weight decay
```

## LR Schedule

Cosine schedule with linear warmup:
```
if step < warmup_steps:
    lr = base_lr * step / warmup_steps
else:
    lr = base_lr * 0.5 * (1 + cos(π * step / total_steps))
```
