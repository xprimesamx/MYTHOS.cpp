# `finetune.cpp` — Fine-Tuning Implementation

**Path:** `src/finetune.cpp`

Fine-tuning support: full fine-tune and LoRA adaptation.

## FineTune Methods

| Method | Description |
|--------|-------------|
| `full_finetune()` | Update all parameters (higher cost) |
| `lora_finetune()` | Low-Rank Adaptation (lower cost) |

## LoRA (Low-Rank Adaptation)

```
W' = W + BA
where:
  W ∈ ℝ^(d×k)  frozen original weights
  B ∈ ℝ^(d×r)  trainable low-rank matrix
  A ∈ ℝ^(r×k)  trainable low-rank matrix
  r << min(d, k) (typically r=8 or r=16)
```

## Freezing

```cpp
// Freeze specified layers by name
finetune(model, {
    .freeze_layers = {"tok_embeddings", "norm", "lm_head"},
    .lora_rank = 8,
    .learning_rate = 1e-5,
});
```
