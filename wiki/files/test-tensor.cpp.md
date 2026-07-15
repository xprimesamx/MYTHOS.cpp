# `test_tensor.cpp` — Tensor Tests

**Path:** `tests/test_tensor.cpp`

Unit tests for the Tensor class.

## Test Cases

| Test | Description |
|------|-------------|
| Construction | Create tensors with various shapes and dtypes |
| Shape ops | `view()`, `reshape()`, `slice()`, `transpose()` |
| Data access | Indexing, iterators, raw pointer access |
| Fill/Zero | `fill()`, `zero_()`, `clone()` |
| Factory | `zeros()`, `ones()`, `arange()` |
| Autograd | `requires_grad`, gradient tracking |
| Memory | Allocation, alignment, copy |

## Coverage

- All tensor constructors
- Shape manipulation (valid and invalid cases)
- Element-wise operations
- Autograd integration
- Edge cases: empty tensors, rank-0, rank-1, rank-4
