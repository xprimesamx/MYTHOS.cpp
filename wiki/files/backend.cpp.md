# `backend.cpp` — Backend Implementation

**Path:** `src/backend.cpp`

Device backend implementation: CPU/CUDA detection and management.

## Backend Singleton

```cpp
Backend& Backend::instance() {
    static Backend backend;
    return backend;
}
```

## Functions

| Function | Description |
|----------|-------------|
| `set_device(DeviceType)` | Switch active device |
| `device()` | Get current device |
| `is_cuda_available()` | Check CUDA availability |
| `synchronize()` | Wait for device operations |

## Device Types

```cpp
enum class DeviceType { CPU, CUDA, AUTO };

// AUTO = try CUDA first, fallback to CPU
```

## Thread Safety

- Singleton access is thread-safe (C++11 static init)
- Device switching is NOT thread-safe
- Call `set_device()` before spawning worker threads
