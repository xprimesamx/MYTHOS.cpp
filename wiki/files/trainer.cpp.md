# `trainer.cpp` — Training Loop

**Path:** `src/trainer.cpp`

Training orchestration: high-level training loop, logging, and checkpoint management.

## Key Functions

| Function | Description |
|----------|-------------|
| `train_model()` | Main training entry point |
| `train_epoch()` | Single epoch training pass |
| `compute_loss()` | Cross-entropy loss computation |
| `backward_pass()` | Gradient computation and optimization |

## Training Loop

```
for each epoch:
    for each batch:
        1. Forward pass → logits
        2. Compute cross-entropy loss
        3. Backward pass (autograd)
        4. Gradient clipping
        5. Optimizer step (AdamW)
        6. Log metrics
    Save checkpoint
```

## Loss Computation

Standard cross-entropy loss:
```
loss = -mean(log(softmax(logits)) * targets)
```
