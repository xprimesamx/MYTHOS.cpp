# `backend.h` — Backend Management

**Path:** `include/oil/backend.h`

Device backend management for CPU, CUDA, and other compute backends.

## Enums & Classes

```cpp
enum class DeviceType { CPU, CUDA, AUTO };

class Backend {
    DeviceType type;
    
    static Backend& instance();
    DeviceType device() const;
    void set_device(DeviceType dev);
    bool is_cuda_available() const;
};
```

### Device Selection

| Device | Description |
|--------|-------------|
| `CPU` | Force CPU execution (always available) |
| `CUDA` | Use CUDA GPU if available |
| `AUTO` | Auto-detect best available device |

### Backend-Specific Operations

Each backend provides:
- Memory allocation
- Tensor operations
- Kernel dispatch
- Device synchronization
