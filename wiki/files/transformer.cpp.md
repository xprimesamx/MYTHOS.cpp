# `transformer.cpp` — Transformer Implementation

**Path:** `src/transformer.cpp`

Implements all transformer architecture components declared in `transformer.h`.

## Implementations

### Embedding
- Token-to-vector lookup via index
- Optional weight tying with LM head

### Linear
- Weight matrix forward pass
- Optional bias addition
- Supports quantized weights (dequantize on-the-fly)

### RMSNorm
- `rms_norm(x) = x * weight / sqrt(mean(x²) + eps)`
- Numerically stable with FP32 accumulator

### Attention
- QKV projection computation
- Scaled dot-product attention
- RoPE application before attention
- KV-cache integration for generation
- Optional causal masking

### FeedForward (SwiGLU)
- Gate projection: `silu(gate(x))`
- Up projection: `up(x)`
- Down projection: `down(gate(x) ⊙ up(x))`
- 8/3 × hidden_size intermediate (standard SwiGLU ratio)

### TransformerBlock
- Pre-norm architecture
- Residual connections with gradient flow
- Full forward pass orchestration
