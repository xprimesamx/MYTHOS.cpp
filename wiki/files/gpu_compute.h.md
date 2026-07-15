# `gpu_compute.h` — GPU Compute

**Path:** `include/oil/gpu_compute.h`

CUDA GPU compute support for accelerated operations.

## GPUCompute Class

```cpp
class GPUCompute {
    static bool is_available();
    static int device_count();
    static std::string device_name(int id = 0);
    static size_t device_memory(int id = 0);
    
    static void synchronize();
    static void set_device(int id);
};
```

### GPU Kernel Operations

| Operation | Description |
|-----------|-------------|
| `matmul_gpu` | Matrix multiplication on GPU |
| `rms_norm_gpu` | RMS normalization |
| `softmax_gpu` | Softmax |
| `rope_gpu` | Rotary Position Embedding |
| `silu_gpu` | SiLU activation |
| `add_gpu` | Element-wise addition |

### Memory Management

| Function | Description |
|----------|-------------|
| `allocate_gpu(size)` | GPU memory allocation |
| `copy_to_gpu(src, dst, size)` | Host → Device transfer |
| `copy_from_gpu(src, dst, size)` | Device → Host transfer |
| `free_gpu(ptr)` | GPU memory deallocation |

### Requirements

- NVIDIA GPU with compute capability ≥ 7.0
- CUDA Toolkit ≥ 11.0
- Build with `-DOIL_GPU=ON`
