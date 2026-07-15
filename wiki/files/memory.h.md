# `memory.h` — Memory Management

**Path:** `include/oil/memory.h`

Memory allocation and management layer with support for pinned memory and GPU buffers.

## Buffer Class

```cpp
class Buffer {
    void* ptr;
    size_t size;
    DeviceType device;
    
    Buffer(size_t size, DeviceType device = DeviceType::CPU);
    ~Buffer();
    
    void* data();
    const void* data() const;
    size_t size() const;
    DeviceType device() const;
};
```

## Memory Functions

| Function | Description |
|----------|-------------|
| `allocate(size, device)` | Allocate memory on device |
| `deallocate(ptr)` | Free memory |
| `copy(src, dst, size)` | Copy between devices (CPU↔CPU, CPU↔GPU) |
| `memset(ptr, val, size)` | Set memory region |
| `is_aligned(ptr, alignment)` | Check alignment |

## Memory Pool

The memory module uses a simple pool allocator to reduce allocation overhead for small tensors.
