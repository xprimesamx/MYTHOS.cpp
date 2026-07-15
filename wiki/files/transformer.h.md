# `transformer.h` — Transformer Architecture

**Path:** `include/oil/transformer.h`

Implements the transformer architecture building blocks: attention, feed-forward, embeddings, and the full transformer block.

## Classes

### Embedding
```cpp
class Embedding {
    Tensor weight;
    Tensor forward(const Tensor& input);
};
```
Token embedding lookup table. Maps token IDs to dense vectors.

### Linear
```cpp
class Linear {
    Tensor weight;
    Tensor bias;  // optional
    Tensor forward(const Tensor& input);
};
```
Fully connected linear layer with optional bias.

### RMSNorm
```cpp
class RMSNorm {
    Tensor weight;
    float eps;
    Tensor forward(const Tensor& input);
};
```
Root Mean Square Layer Normalization — faster than LayerNorm with comparable quality.

### Attention
```cpp
class Attention {
    Linear q_proj, k_proj, v_proj, o_proj;
    int num_heads, head_dim;
    Tensor forward(const Tensor& x, KVCache* cache);
};
```
Multi-head self-attention with:
- Separate Q, K, V projections
- Rotary Position Embedding
- Optional KV caching for efficient generation
- Flash attention compatible

### FeedForward
```cpp
class FeedForward {
    Linear gate_proj, up_proj, down_proj;
    Tensor forward(const Tensor& x);
};
```
SwiGLU feed-forward network: `down(gate(x) ⊙ silu(up(x)))`.

### TransformerBlock
```cpp
class TransformerBlock {
    Attention attn;
    FeedForward ffn;
    RMSNorm norm1, norm2;
    Tensor forward(const Tensor& x, KVCache* cache = nullptr);
};
```
Single transformer layer: `x + attn(norm1(x)) + ffn(norm2(x))`.

## Architecture

Each transformer block applies:
1. Pre-normalization (RMSNorm)
2. Multi-head self-attention with residual connection
3. Pre-normalization (RMSNorm)
4. SwiGLU feed-forward with residual connection
