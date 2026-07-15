# `gpu_extras.cpp` — GPU Extras Implementation

**Path:** `src/gpu_extras.cpp`

Additional GPU utilities: device info queries, error checking, and performance timing.

## Functions

| Function | Description |
|----------|-------------|
| `get_device_info(id)` | Query device properties |
| `get_memory_usage()` | Free/total memory |
| `check_cuda_error()` | Check and log CUDA errors |
| `get_kernel_time()` | CUDA event-based timing |

## Error Checking

```cpp
#define CUDA_CHECK(call) do {
    cudaError_t err = call;
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n",
                __FILE__, __LINE__, cudaGetErrorString(err));
        exit(err);
    }
} while(0)
```

## Timing

```cpp
GPUTimer timer;
timer.start();
matmul_gpu(...);  // kernel launch
timer.stop();
float ms = timer.elapsed_ms();  // ~0.1ms precision
```
