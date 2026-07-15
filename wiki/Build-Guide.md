# Build Guide

## Quick Start

```bash
# Configure
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build all
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure
```

## Platform-Specific Instructions

### Windows (Clang-cl)

```bash
# Prerequisites:
# 1. Install Visual Studio 2022 with "Desktop development with C++"
# 2. Install LLVM/Clang (clang-cl)
# 3. Install Ninja

# Open "Developer Command Prompt for VS 2022"
# Then:
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=clang-cl ^
    -DCMAKE_CXX_COMPILER=clang-cl
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

**Verified build:** Windows 11 + Clang 22.1.7 (clang-cl) — ✅ All 18 executables build, 9/9 tests pass

### Linux (GCC)

```bash
# Prerequisites: g++ ≥ 12, cmake ≥ 3.24, ninja
sudo apt install build-essential cmake ninja-build

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### macOS (Apple Clang)

```bash
brew install cmake ninja
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Debug | `Release`, `Debug`, `RelWithDebInfo` |
| `OIL_GPU` | OFF | Enable CUDA GPU support |
| `OIL_TEST` | ON | Build test suite |
| `OIL_BENCH` | ON | Build benchmarks |
| `OIL_PYTHON` | OFF | Build Python bindings |
| `OIL_AVX2` | AUTO | Force AVX2 on/off (AUTO = detect) |

## Build Targets

```bash
# Build everything
cmake --build build --parallel

# Specific targets
cmake --build build --target oil_core      # Core library only
cmake --build build --target oil_tensor    # Tensor only
cmake --build build --target oil_math      # Math library
cmake --build build --target oil-infer     # Inference CLI
cmake --build build --target test_tensor   # Tensor tests only

# Clean build
cmake --build build --target clean
```

## Build Output (18 Libraries + 6 Tools + 17 Tests)

### Libraries
| Target | Description |
|--------|-------------|
| `oil_core` | Tensor, Autograd, Memory, Types, Random |
| `oil_math` | Math ops (AVX2 + fallback) |
| `oil_kernel` | Quantization/dequantization kernels |
| `oil_model` | Model & Transformer |
| `oil_format` | OIL format I/O |
| `oil_backend` | Device backend management |
| `oil_engine` | Inference engine |
| `oil_trainer` | Training engine |
| `oil_gpu` | CUDA GPU acceleration |
| `oil_moe` | Mixture-of-Experts |
| `oil_oil8` | OIL8 codec |
| `oil_dense` | Dense training |
| `oil_tokenizer` | BPE tokenizer |
| `oil_asi` | ASI module |
| `oil_infer`, `oil_convert`, `oil_train`, `oil_info` | Tool libraries |

### Tools
| Target | Description |
|--------|-------------|
| `oil-convert` | Model conversion (safetensors → OIL) |
| `oil-train` | Train from scratch |
| `oil-infer` | Run inference |
| `oil-finetune` | Fine-tune models |
| `oil-info` | Model metadata |
| `oil-bench` | Performance benchmarks |

## Tests

```bash
# Run all
ctest --test-dir build --output-on-failure

# Run specific
ctest -R test_tensor -V
ctest -R test_training -V
ctest -R test_format -V
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `clang-cl not found` | Open "Developer Command Prompt for VS 2022" |
| `AVX2 not supported` | Auto-falls back to C++20 kernels |
| `CUDA not found` | Set `-DOIL_GPU=OFF` |
| `Ninja not found` | Install ninja or use `-G "Visual Studio 17 2022"` |
| Link errors in Debug | Try Release build |
| Test crashes in Debug | ASAN/UBsan may detect issues; use Release |
