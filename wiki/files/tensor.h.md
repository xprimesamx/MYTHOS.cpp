# `tensor.h` — Multi-Dimensional Tensor

**Path:** `include/oil/tensor.h`

The core data structure — a multi-dimensional array with shape, dtype, and optional autograd support.

## Tensor Class

### Construction

| Constructor | Description |
|-------------|-------------|
| `Tensor()` | Empty tensor |
| `Tensor(Shape, DType)` | Allocate tensor with given shape and dtype |
| `Tensor(Shape, shared_ptr<Buffer>, DType)` | Wrap existing buffer |

### Accessors

| Method | Description |
|--------|-------------|
| `shape()` | Returns const `Shape&` |
| `dim(int i)` | Size of i-th dimension |
| `rank()` | Number of dimensions |
| `numel()` | Total number of elements |
| `dtype()` | Element data type |
| `data()` | Raw void pointer to data |
| `data<T>()` | Typed pointer |
| `size_bytes()` | Total size in bytes |

### View Operations

| Method | Description |
|--------|-------------|
| `view(Shape)` | New view with different shape (shares data) |
| `slice(dim, start, end)` | Slice along dimension |
| `reshape(Shape)` | Reshape (may copy if needed) |
| `transpose(dim1, dim2)` | Transpose two dimensions |

### Data Operations

| Method | Description |
|--------|-------------|
| `fill(float)` | Fill with scalar value |
| `copy_from(Tensor)` | Copy data from source |
| `copy_to(Tensor)` | Copy data to destination |
| `clone()` | Deep copy |
| `zero_()` | Zero out elements |

### Factory Methods

| Method | Description |
|--------|-------------|
| `zeros(Shape)` | Create zero-filled tensor |
| `ones(Shape)` | Create one-filled tensor |
| `arange(n)` | Create 1D tensor `[0, 1, ..., n-1]` |

### Autograd Support

| Method | Description |
|--------|-------------|
| `requires_grad()` | Check if gradient tracking is enabled |
| `requires_grad(bool)` | Enable/disable gradient tracking |
| `grad()` | Access gradient tensor |

### Operators

| Operator | Description |
|----------|-------------|
| `operator[]` | Index into tensor (returns element or sub-tensor) |
| Arithmetic operators | Element-wise `+`, `-`, `*`, `/` with autograd |

## Usage Example

```cpp
auto tensor = Tensor::zeros({2, 3, 64});
tensor.fill(1.0f);
auto reshaped = tensor.reshape({6, 64});
auto slice = tensor.slice(0, 0, 1);
```
