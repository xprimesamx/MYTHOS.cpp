#pragma once

#include <cstdint>
#include <cstddef>

namespace oil {

class RNG {
public:
    explicit RNG(uint64_t seed = 42);

    void seed(uint64_t s);

    float uniform();
    float normal();
    int uniform_int(int lo, int hi);

    uint64_t next_u64();
    uint32_t next_u32();

private:
    uint64_t s[2];

    static uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

} // namespace oil
