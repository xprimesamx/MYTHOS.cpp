# `autograd.cpp` — Autograd Engine Implementation

**Path:** `src/autograd.cpp`

Implements the automatic differentiation engine — dynamic computation graph building and backward pass.

## Key Implementations

| Feature | Implementation |
|---------|---------------|
| Computation graph | Dynamic DAG built during forward |
| Gradient functions | Lambda captures storing operation context |
| Backward pass | Topological traversal with gradient accumulation |
| Gradient accumulation | Sums gradients from multiple paths |
| Graph disposal | Freed after backward to save memory |

## Gradient Registration

Each math operation registers its gradient:
- `matmul` → dA = dC × Bᵀ, dB = Aᵀ × dC
- `add` → passes gradient to both inputs
- `mul` → dA = dC * B, dB = A * dC
- `silu` → dA = dC * (sigmoid(A) + A * sigmoid(A) * (1 - sigmoid(A)))
- `softmax` → dA = softmax(A) * (dC - sum(dC * softmax(A)))
- `rms_norm` → chain rule through normalization
