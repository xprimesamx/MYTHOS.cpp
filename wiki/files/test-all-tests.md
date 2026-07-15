# Test Files — Complete Reference

## All Tests

| File | Area | Key Tests |
|------|------|-----------|
| `test_all.cpp` | Integration | Full pipeline: tensor→model→load→save→train→infer |
| `test_tensor.cpp` | Tensor | Construction, shape ops, fill/clone, autograd |
| `test_math.cpp` | Math | Matmul, rms_norm, softmax, rope, activations |
| `test_model.cpp` | Model | Load/save `.oil`, forward pass, param count |
| `test_kernel.cpp` | Kernels | Quantize/dequantize (OIL4, OIL8, binary, ternary) |
| `test_format.cpp` | Format | OIL read/write, header, config, tensor metadata |
| `test_tokenizer.cpp` | Tokenizer | Encode, decode, special tokens, edge cases |
| `test_gpu.cpp` | GPU | Device info, memory, kernel launch |
| `test_trainer.cpp` | Trainer | Training loop, loss computation |
| `test_training.cpp` | E2E training | Train small model on tiny data |
| `test_e2e_training.cpp` | Full pipeline | Convert→train→infer→eval cycle |
| `test_bench.cpp` | Benchmarks | Benchmark validation |
| `test_debug.cpp` | Debug | Debug utilities |
| `test_save_load_debug.cpp` | Save/Load | Save/load corner cases |
| `test_integration.cpp` | Integration | Cross-module interaction tests |
| `_debug_embed.cpp` | Debug embed | Debug embedding helper |
| `test_trainer.cpp` | Trainer extensions | Additional trainer tests |

## Running Tests

```bash
# All tests
ctest --test-dir build --output-on-failure

# Individual
ctest --test-dir build -R test_tensor -V
ctest --test-dir build -R test_training -V

# Build a test target
cmake --build build --target test_tensor && build/tests/test_tensor
```
