# `gpu_extras.h` — GPU Extra Utilities

**Path:** `include/oil/gpu_extras.h`

Additional GPU utilities: device info, memory queries, and performance monitoring.

## GPUDeviceInfo

```cpp
struct GPUDeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
};
```

## GPU Utilities

| Function | Description |
|----------|-------------|
| `get_device_info(int id)` | Get detailed GPU info |
| `get_memory_usage()` | Current memory utilization |
| `get_device_count()` | Number of CUDA devices |
| `synchronize_device()` | Wait for GPU operations |
| `check_cuda_error()` | Check and report CUDA errors |
| `get_kernel_time()` | Kernel execution timing |

## Timing

```cpp
class GPUTimer {
    GPUTimer();
    void start();
    void stop();
    float elapsed_ms() const;
};
```

CUDA event-based timing for GPU kernel performance measurement.
