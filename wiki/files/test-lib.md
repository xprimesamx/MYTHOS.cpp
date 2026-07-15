# Test Suite Overview

## All Test Files

| File | Description |
|------|-------------|
| [test_all.cpp](test_all.cpp.md) | Comprehensive integration test |
| [test_tensor.cpp](test-tensor.cpp.md) | Tensor operations |
| [test_math.cpp](test_math.cpp.md) | Math operations |
| [test_model.cpp](test_model.cpp.md) | Model load/save/forward |
| [test_kernel.cpp](test_kernel.cpp.md) | Quantization kernels |
| [test_format.cpp](test_format.cpp.md) | OIL format I/O |
| [test_tokenizer.cpp](test_tokenizer.cpp.md) | BPE tokenization |
| [test_gpu.cpp](test_gpu.cpp.md) | GPU operations |
| [test_trainer.cpp](test_trainer.cpp.md) | Training loop |
| [test_training.cpp](test_training.cpp.md) | End-to-end training |
| [test_e2e_training.cpp](test_e2e_training.cpp.md) | Full pipeline test |
| [test_bench.cpp](test_bench.cpp.md) | Benchmark validation |
| [test_debug.cpp](test_debug.cpp.md) | Debug utilities |
| [test_save_load_debug.cpp](test_save_load_debug.cpp.md) | Save/load testing |
| [test_integration.cpp](test_integration.cpp.md) | Integration tests |
| [_debug_embed.cpp](test-debug-embed.cpp.md) | Debug embedding |

## Build & Run

```bash
# Build and run all tests
cmake -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure

# Run specific test
ctest --test-dir build -R test_tensor --output-on-failure
```
