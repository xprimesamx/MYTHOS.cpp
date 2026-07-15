# `tensor.cpp` — Tensor Implementation

**Path:** `src/tensor.cpp`

Implements the `Tensor` class: construction, operations, and memory management.

## Key Implementations

| Operation | Description |
|-----------|-------------|
| Tensor constructors | Allocation, shape validation, dtype setup |
| `view()` / `reshape()` | Shape manipulation with data sharing |
| `slice()` | Dimension slicing |
| `transpose()` | Dimension permutation |
| `fill()` | Element-wise fill |
| `clone()` | Deep copy implementation |
| `copy_from()` / `copy_to()` | Inter-tensor data transfer |
| Factory methods | `zeros()`, `ones()`, `arange()` |
| Indexing | `operator[]` with bounds checking |

## Autograd Integration

- Tracks `requires_grad` flag
- Maintains gradient buffer
- Registers gradient functions on operations
- Supports backward pass propagation
