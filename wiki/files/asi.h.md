# `asi.h` — Advanced Synthetic Intelligence

**Path:** `include/oil/asi.h`

ASI (Advanced Synthetic Intelligence) module — experimental high-level cognitive architecture.

## ASI Classes

```cpp
class ASI {
    ASIConfig config;
    std::unique_ptr<Model> model;
    std::unique_ptr<MoE> moe;
    
    ASI(const ASIConfig& cfg);
    
    Tensor think(const Tensor& input);
    Tensor reason(const Tensor& context);
    void learn(const Tensor& experience);
    ASIState get_state() const;
};
```

### ASIConfig

```cpp
struct ASIConfig {
    int reasoning_depth = 4;
    int num_candidates = 8;
    float temperature = 0.7f;
    bool use_moe = true;
    bool use_reflection = true;
    // ... experimental parameters
};
```

### Key Capabilities

| Capability | Description |
|------------|-------------|
| `think()` | Generate thoughts given input |
| `reason()` | Multi-step reasoning chain |
| `learn()` | Incorporate new experiences |
| `reflect()` | Self-reflection on outputs |

### ASI vs Standard Models

The ASI module extends standard transformer inference with:
- Multi-step reasoning before output
- MoE-enhanced thought generation
- Self-reflection/self-correction loops
- Experience accumulation across episodes
