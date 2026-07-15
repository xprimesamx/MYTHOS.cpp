# `asi_pipeline.h` — ASI Pipeline

**Path:** `include/oil/asi_pipeline.h`

Advanced Synthetic Intelligence pipeline — multi-stage processing for enhanced reasoning.

## ASIPipeline

```cpp
class ASIPipeline {
    ASIConfig config;
    std::unique_ptr<Model> model;
    std::unique_ptr<MoE> moe;
    
    ASIPipeline(const ASIConfig& cfg);
    
    Tensor process(const Tensor& input);
    Tensor think_deep(const Tensor& context, int depth);
    Tensor reflect(const Tensor& thought, const Tensor& context);
    Tensor synthesize(const std::vector<Tensor>& thoughts);
};
```

### Pipeline Stages

| Stage | Description |
|-------|-------------|
| 1. **Input Processing** | Tokenize and embed input |
| 2. **Deep Thinking** | Multi-step reasoning chain |
| 3. **Reflection** | Self-critique and refinement |
| 4. **Synthesis** | Combine multiple reasoning paths |
| 5. **Output Generation** | Final response generation |

### Pipeline Flow

```
Input → Tokenize → Think(step 1) → Think(step N) → Reflect → Synthesize → Output
                         ↓                                      ↑
                   MoE enhances                          Multiple paths combined
```

### MoE Integration

Each thinking step can route through MoE for specialized reasoning across different expert domains.
