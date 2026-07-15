# `moe_variants.cpp` — MoE Variants Implementation

**Path:** `src/moe_variants.cpp`

Different MoE architecture variants and routing strategies.

## Implementations

### Switch Transformer
- Router routes to exactly 1 expert (top-1)
- Higher capacity per expert
- Simpler routing, worse quality than top-2

### Top-2 Gating (Default)
- Router routes to top-2 experts
- Better quality than Switch
- Standard MoE approach

### Expert Choice (Future)
- Experts choose tokens instead of tokens choosing experts
- Better load balancing
- Not yet implemented

## Config

```cpp
// In TransformerConfig:
config.use_moe = true;
config.num_experts = 8;
config.top_k = 2;  // Switch = 1, Default = 2
```
