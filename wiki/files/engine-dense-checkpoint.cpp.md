# `engines/trainer/dense/checkpoint.cpp` — Checkpointing

**Path:** `engines/trainer/dense/checkpoint.cpp`

Training checkpoint save/load for resume and model selection.

## Checkpoint Functions

| Function | Description |
|----------|-------------|
| `save(path, model, optimizer, step)` | Full training state save |
| `load(path, model, optimizer, step)` | Full training state restore |
| `save_best(path, model, step, metric)` | Save best model by metric |

## Checkpoint Contents

```cpp
struct Checkpoint {
    Model weights;         // .oil format
    OptimizerState opt;    // AdamW moments
    int step;              // Training step
    float best_loss;       // Best validation loss
    // ... metadata
};
```

## File Naming

```
checkpoints/
├── step_1000.oil       # At step 1000 (model only)
├── step_1000.opt       # Optimizer state at step 1000
├── step_2000.oil       # At step 2000
├── step_2000.opt       # Optimizer state at step 2000
└── best.oil            # Best model by validation loss
```
