#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <cstdlib>
#include <new>

namespace oil {

struct AlignedAllocator {
    static void* allocate(size_t bytes, size_t alignment = 64) {
#ifdef _WIN32
        void* ptr = _aligned_malloc(bytes, alignment);
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, bytes) != 0)
            ptr = nullptr;
#endif
        if (!ptr && bytes)
            throw std::bad_alloc();
        return ptr;
    }

    static void deallocate(void* ptr) {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
};

class Buffer {
public:
    Buffer() = default;

    explicit Buffer(size_t size_bytes, size_t alignment = 64)
        : block_{allocate_block(size_bytes, alignment)} {}

    Buffer(const Buffer& other)
        : block_{other.block_} {
        if (block_.refcount)
            block_.refcount->fetch_add(1, std::memory_order_acq_rel);
    }

    Buffer& operator=(const Buffer& other) {
        if (this == &other)
            return *this;
        release();
        block_ = other.block_;
        if (block_.refcount)
            block_.refcount->fetch_add(1, std::memory_order_acq_rel);
        return *this;
    }

    ~Buffer() {
        release();
    }

    void* data() const { return block_.ptr; }

    size_t size() const { return block_.size; }

    bool empty() const { return block_.size == 0; }

    void resize(size_t new_size) {
        if (block_.ptr) {
            if (block_.refcount->load(std::memory_order_acquire) > 1)
                *this = Buffer(new_size, 64);
            else
                *this = Buffer(new_size, 64);
        } else {
            *this = Buffer(new_size, 64);
        }
    }

    Buffer slice(size_t offset, size_t count) const {
        if (offset + count > block_.size)
            return Buffer();
        Buffer sub;
        sub.block_.ptr = static_cast<char*>(block_.ptr) + offset;
        sub.block_.refcount = block_.refcount;
        sub.block_.size = count;
        if (sub.block_.refcount)
            sub.block_.refcount->fetch_add(1, std::memory_order_acq_rel);
        return sub;
    }

private:
    struct Block {
        void* ptr;
        std::atomic<int>* refcount;
        size_t size;
    };

    Block block_{nullptr, nullptr, 0};

    static Block allocate_block(size_t bytes, size_t alignment) {
        if (bytes == 0) return {nullptr, nullptr, 0};
        void* ptr = AlignedAllocator::allocate(bytes, alignment);
        auto* rc = new std::atomic<int>(1);
        return {ptr, rc, bytes};
    }

    void release() {
        if (block_.refcount &&
            block_.refcount->fetch_sub(1, std::memory_order_acq_rel) == 1) {
            AlignedAllocator::deallocate(block_.ptr);
            delete block_.refcount;
        }
        block_ = {nullptr, nullptr, 0};
    }
};

class MemoryPool {
public:
    explicit MemoryPool(size_t total_size)
        : capacity_(total_size), offset_(0) {
        data_ = static_cast<char*>(AlignedAllocator::allocate(total_size));
    }

    ~MemoryPool() {
        AlignedAllocator::deallocate(data_);
    }

    void* allocate(size_t bytes) {
        size_t aligned = (bytes + 63) & ~size_t(63);
        if (offset_ + aligned > capacity_)
            return nullptr;
        void* ptr = data_ + offset_;
        offset_ += aligned;
        return ptr;
    }

    void reset() {
        offset_ = 0;
    }

private:
    char* data_;
    size_t capacity_;
    size_t offset_;
};

} // namespace oil
