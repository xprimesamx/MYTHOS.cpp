# `memory.cpp` — Memory Management Implementation

**Path:** `src/memory.cpp`

Memory allocation and management: Buffer class, pooling, and cross-device copies.

## Buffer Lifecycle

```cpp
Buffer::Buffer(size, device) {
    // Allocate: malloc for CPU, cudaMalloc for GPU
    // Track size and device for deallocation
}

Buffer::~Buffer() {
    // Free: free() for CPU, cudaFree for GPU
}
```

## Copy Operations

| Operation | Implementation |
|-----------|---------------|
| CPU ↔ CPU | `memcpy` |
| CPU → GPU | `cudaMemcpyHostToDevice` |
| GPU → CPU | `cudaMemcpyDeviceToHost` |
| GPU → GPU | `cudaMemcpyDeviceToDevice` |

## Memory Pool

Simple pool allocator to reduce malloc overhead:
- Fixed-size blocks (64KB)
- Free list for reuse
- Grows on demand
- Thread-local for thread safety
