#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>

namespace oil {

enum class Format : uint8_t {
    BINARY  = 0, // 1 bit, {-1, +1}
    TERNARY = 1, // 1.58 bits, {-1, 0, +1}
    OIL4    = 2, // 4 bits, codebook 16 × FP16
    OIL8    = 3, // 8 bits, codebook 256 × FP32
    FP16    = 4, // 16 bits, native half
    FP32    = 5, // 32 bits, native float
};

inline const char* format_name(Format f) {
    switch (f) {
        case Format::BINARY:  return "binary";
        case Format::TERNARY: return "ternary";
        case Format::OIL4:    return "oil4";
        case Format::OIL8:    return "oil8";
        case Format::FP16:    return "fp16";
        case Format::FP32:    return "fp32";
    }
    return "unknown";
}

inline float format_bpw(Format f) {
    switch (f) {
        case Format::BINARY:  return 1.0f;
        case Format::TERNARY: return 1.58f;
        case Format::OIL4:    return 4.0f;
        case Format::OIL8:    return 8.0f;
        case Format::FP16:    return 16.0f;
        case Format::FP32:    return 32.0f;
    }
    return 0;
}

enum class DType : uint8_t {
    U8,      // uint8_t
    U4,      // 4-bit packed (2 per byte)
    I2,      // 2-bit ternary packed (4 per byte)
    I1,      // 1-bit binary packed (8 per byte)
    F16,     // half precision
    F32,     // single precision
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::U8: return 1;
        case DType::U4: return 1; // 2 per byte
        case DType::I2: return 1; // 4 per byte
        case DType::I1: return 1; // 8 per byte
        case DType::F16: return 2;
        case DType::F32: return 4;
    }
    return 0;
}

inline DType format_to_dtype(Format f) {
    switch (f) {
        case Format::BINARY:  return DType::I1;
        case Format::TERNARY: return DType::I2;
        case Format::OIL4:    return DType::U4;
        case Format::OIL8:    return DType::U8;
        case Format::FP16:    return DType::F16;
        case Format::FP32:    return DType::F32;
    }
    return DType::F32;
}

struct Shape {
    int64_t dims[8];
    int rank;

    Shape() : rank(0) {}
    explicit Shape(int64_t d0) : rank(1) { dims[0]=d0; }
    Shape(int64_t d0, int64_t d1) : rank(2) { dims[0]=d0; dims[1]=d1; }
    Shape(int64_t d0, int64_t d1, int64_t d2) : rank(3) { dims[0]=d0; dims[1]=d1; dims[2]=d2; }
    Shape(std::initializer_list<int64_t> l) : rank((int)l.size()) {
        int i=0; for (auto x: l) dims[i++] = x;
    }

    int64_t& operator[](int i) { return dims[i]; }
    const int64_t& operator[](int i) const { return dims[i]; }

    int64_t numel() const {
        int64_t n = 1;
        for (int i=0; i<rank; i++) n *= dims[i];
        return n;
    }

    bool operator==(const Shape& o) const {
        if (rank != o.rank) return false;
        for (int i=0; i<rank; i++) if (dims[i] != o.dims[i]) return false;
        return true;
    }

    bool operator!=(const Shape& o) const { return !(*this == o); }

    std::string to_string() const {
        std::string s = "[";
        for (int i=0; i<rank; i++) {
            if (i) s += ",";
            s += std::to_string(dims[i]);
        }
        s += "]";
        return s;
    }
};

struct Status {
    bool ok;
    std::string msg;
    Status() : ok(true) {}
    Status(const std::string& e) : ok(false), msg(e) {}
    static Status error(const std::string& m) { return Status(m); }
    static Status success() { return Status(); }
    explicit operator bool() const { return ok; }
};

struct Config {
    int num_threads = 1;
    uint64_t seed = 42;
    size_t pool_size = 64 * 1024 * 1024; // 64MB temp pool
    bool verbose = false;
};

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

#define OIL_CHECK(cond, msg) \
    do { if (!(cond)) throw oil::Error(msg); } while(0)

} // namespace oil
