# `autograd.h` — Automatic Differentiation

**Path:** `include/oil/autograd.h`

Automatic differentiation engine supporting dynamic computation graphs.

## Autograd Engine

```cpp
class AutogradEngine {
    static AutogradEngine& instance();
    
    Tensor forward(Tensor& input, GradientFn fn);
    void backward(Tensor& loss);
    void zero_grad();
};
```

### Key Concepts

| Component | Description |
|-----------|-------------|
| **GradientFn** | `std::function<void()>` — stores how to compute gradients |
| **Computation Graph** | Built dynamically during forward pass |
| **Backward Pass** | Traverses graph in topological order |

### How It Works

1. Each tensor operation registers a `GradientFn` that captures operands
2. On `backward()`, the engine traverses the graph from loss backwards
3. Gradients are accumulated in each tensor's `grad()` buffer
4. Graph is discarded after backward (dynamic)

### Operations with Autograd

The engine supports automatic differentiation for:
- Matrix multiplication
- Element-wise operations (add, mul, sub, div)
- Activation functions (silu, gelu, relu, tanh)
- Normalization (rms_norm)
- Softmax
- View/reshape operations
