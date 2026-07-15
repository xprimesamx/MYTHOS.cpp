# `gpu_compute.cpp` — GPU Compute Implementation

**Path:** `src/gpu_compute.cpp`

CUDA GPU compute: device management and kernel launches.

## Functions

| Function | Description |
|----------|-------------|
| `is_available()` | Check CUDA runtime availability |
| `device_count()` | Number of CUDA-capable devices |
| `device_name(id)` | Human-readable device name |
| `synchronize()` | Wait for all pending CUDA operations |

## Kernel Launches

```cpp
// Forward operation dispatch
if (Backend::instance().device() == DeviceType::CUDA) {
    matmul_gpu(C, A, B);
} else {
    matmul_cpu(C, A, B);
}
```

## Memory Transfers

| Function | Description |
|----------|-------------|
| `allocate_gpu(size)` | `cudaMalloc` wrapper |
| `copy_to_gpu(src, dst, size)` | `cudaMemcpy` H→D |
| `copy_from_gpu(src, dst, size)` | `cudaMemcpy` D→H |
| `free_gpu(ptr)` | `cudaFree` wrapper |
