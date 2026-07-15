# `random.h` — Random Number Generation

**Path:** `include/oil/random.h`

Random number generation utilities for weight initialization and sampling.

## Random Class

```cpp
class Random {
    Random(uint64_t seed = std::random_device{}());
    
    float uniform(float min = 0.0f, float max = 1.0f);
    float normal(float mean = 0.0f, float stddev = 1.0f);
    int64_t uniform_int(int64_t min, int64_t max);
    void shuffle(std::vector<int>& indices);
    uint64_t seed() const;
};
```

### Distributions

| Method | Distribution | Range |
|--------|-------------|-------|
| `uniform(min, max)` | Uniform | [min, max) |
| `normal(mean, stddev)` | Gaussian | unbounded |
| `uniform_int(min, max)` | Discrete uniform | [min, max] |

### Use Cases

- Xavier/Gaussian weight initialization
- Training data shuffling
- Dropout mask generation
- Temperature-based sampling
