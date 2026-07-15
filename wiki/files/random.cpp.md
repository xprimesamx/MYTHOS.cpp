# `random.cpp` — Random Number Generation Implementation

**Path:** `src/random.cpp`

Random number generation using C++20 `<random>` library with PCG/MT19937 engines.

## Implementation

```cpp
Random::Random(uint64_t seed) : rng(seed ? seed : random_device{}()) {}

float Random::uniform(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

float Random::normal(float mean, float stddev) {
    std::normal_distribution<float> dist(mean, stddev);
    return dist(rng);
}

int64_t Random::uniform_int(int64_t min, int64_t max) {
    std::uniform_int_distribution<int64_t> dist(min, max);
    return dist(rng);
}
```

## Distributions

| Distribution | Generator | Use |
|-------------|-----------|-----|
| Uniform real | `std::uniform_real_distribution` | Weight init, dropout |
| Normal | `std::normal_distribution` | Xavier init |
| Uniform int | `std::uniform_int_distribution` | Shuffle, sampling |
