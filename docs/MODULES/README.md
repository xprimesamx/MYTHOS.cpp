# Module Documentation

> **Detailed Documentation for Each MYTHOS.cpp Module**

---

## 📚 Available Module Documentation

This directory contains detailed documentation for each major component of MYTHOS.cpp. Each module document covers:

- Overview and purpose
- Key features
- API reference
- Implementation details
- Best practices
- Common pitfalls
- Examples

---

## 🗂️ Module List

### Core Modules

| Module | Description | Files |
|--------|-------------|-------|
| **[Tensor](tensor.md)** | Multi-dimensional array with autograd support | `tensor.h`, `tensor.cpp` |
| **[Autograd](autograd.md)** | Automatic differentiation engine | `autograd.h`, `autograd.cpp` |
| **[Math](math.md)** | Mathematical operations with AVX2 optimization | `math.h`, `math.cpp`, `math_avx2.cpp` |
| **[Types](types.md)** | Core types, formats, and data structures | `types.h` |
| **[Memory](memory.md)** | Memory management and allocation | `memory.h`, `memory.cpp` |

### Model Modules

| Module | Description | Files |
|--------|-------------|-------|
| **[Transformer](transformer.md)** | Transformer architecture implementation | `transformer.h`, `transformer.cpp` |
| **[Model](model.md)** | Base model class and model management | `model.h`, `model.cpp` |
| **[Backend](backend.md)** | Backend configuration and device management | `backend.h`, `backend.cpp` |

### Quantization Modules

| Module | Description | Files |
|--------|-------------|-------|
| **[Quantization](quantization.md)** | Quantization kernels and formats | `kernel_*.cpp`, `int8_quant.cpp`, etc. |
| **[OIL Format](oil_format.md)** | OIL binary format specification | `oil_format.h`, `oil_format.cpp` |
| **[Codebook](codebook.md)** | Vector quantization codebooks | `codebook.h`, `codebook.cpp` |
| **[STE Quantizer](ste_quantizer.md)** | Straight-Through Estimator for training | `ste_quantizer.h`, `ste_quantizer.cpp` |
| **[Format Planner](format_planner.md)** | Importance-based format allocation | `format_planner.h`, `format_planner.cpp` |

### Training Modules

| Module | Description | Files |
|--------|-------------|-------|
| **[Trainer](trainer.md)** | Training loop and infrastructure | `trainer.h`, `trainer.cpp` |
| **[Optimizer](optimizer.md)** | Optimization algorithms (SGD, Adam, AdamW) | `optimizer.h`, `optimizer.cpp` |
| **[Finetune](finetune.md)** | Fine-tuning implementation | `finetune.h`, `finetune.cpp` |

### Inference Modules

| Module | Description | Files |
|--------|-------------|-------|
| **[Tokenizer](tokenizer.md)** | BPE tokenizer implementation | `tokenizer.h`, `bpe_tokenizer.cpp` |
| **[Sampler](sampler.md)** | Token sampling strategies | `sampler.h`, `sampler.cpp` |
| **[KV Cache](kv_cache.md)** | Key-Value cache for attention | `kv_cache.h`, `kv_cache.cpp` |

### Advanced Modules

| Module | Description | Files |
|--------|-------------|-------|
| **[MoE Variants](moe.md)** | Mixture of Experts implementation | `moe_variants.h`, `moe_variants.cpp` |
| **[GPU Compute](gpu_compute.md)** | DirectX 12 GPU acceleration | `gpu_compute.h`, `gpu_compute.cpp` |

---

## 🎯 Navigation Guide

### For Beginners

Start with these modules to understand the foundation:

1. **[Tensor](tensor.md)** - The fundamental data structure
2. **[Types](types.md)** - Core types and enums
3. **[Memory](memory.md)** - Memory management
4. **[Math](math.md)** - Mathematical operations

### For Intermediate Users

Understand the model implementation:

1. **[Transformer](transformer.md)** - Transformer architecture
2. **[Model](model.md)** - Model management
3. **[Tokenizer](tokenizer.md)** - Tokenization
4. **[Sampler](sampler.md)** - Sampling strategies

### For Advanced Users

Dive into optimization and specialized features:

1. **[Autograd](autograd.md)** - Automatic differentiation
2. **[Quantization](quantization.md)** - Quantization techniques
3. **[OIL Format](oil_format.md)** - Binary format
4. **[Trainer](trainer.md)** - Training infrastructure
5. **[MoE Variants](moe.md)** - Mixture of Experts
6. **[GPU Compute](gpu_compute.md)** - GPU acceleration

---

## 🔗 Module Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                        MODULE DEPENDENCIES                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │   Types      │    │   Memory     │    │    Math      │        │
│  │              │    │              │    │              │        │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘        │
│         │                  │                  │                │
│         ▼                  ▼                  ▼                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                        Tensor                               │    │
│  └───────────────────────────────┬───────────────────────────┘    │
│                                  │                                │
│          ┌───────────────────────┼───────────────────────┐      │
│          │                       │                       │      │
│          ▼                       ▼                       ▼      │
│  ┌──────────────┐       ┌──────────────┐       ┌────────┐   │
│  │  Autograd    │       │  Transformer │       │ Model  │   │
│  └──────┬───────┘       └──────┬───────┘       └────┬───┘   │
│         │                      │                    │         │
│         ▼                      ▼                    ▼         │
│  ┌──────────────┐       ┌──────────────┐   ┌──────────────┐  │
│  │   Trainer    │       │    MoE       │   │   Backend    │  │
│  └──────┬───────┘       └──────────────┘   └──────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Optimizer   │    │   Tokenizer  │    │     KV      │      │
│  └──────────────┘    │              │    │    Cache    │      │
│                     └──────────────┘    └──────────────┘      │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Quant      │    │  OIL Format  │    │  GPU Compute │      │
│  │  Kernels     │    │              │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📖 Documentation Status

| Module | Documentation | Status | Priority |
|--------|--------------|--------|----------|
| Tensor | ✅ Complete | [tensor.md](tensor.md) | High |
| Types | ⏳ Planned | | Medium |
| Memory | ⏳ Planned | | Medium |
| Math | ⏳ Planned | | Medium |
| Autograd | ⏳ Planned | | Medium |
| Transformer | ⏳ Planned | | Medium |
| Model | ⏳ Planned | | Medium |
| Backend | ⏳ Planned | | Low |
| Quantization | ⏳ Planned | | Medium |
| OIL Format | ⏳ Planned | | Medium |
| Codebook | ⏳ Planned | | Low |
| STE Quantizer | ⏳ Planned | | Low |
| Format Planner | ⏳ Planned | | Low |
| Trainer | ⏳ Planned | | Medium |
| Optimizer | ⏳ Planned | | Medium |
| Finetune | ⏳ Planned | | Low |
| Tokenizer | ⏳ Planned | | Medium |
| Sampler | ⏳ Planned | | Medium |
| KV Cache | ⏳ Planned | | Low |
| MoE Variants | ⏳ Planned | | Medium |
| GPU Compute | ⏳ Planned | | Medium |

---

## 🛠️ How to Use This Documentation

### For Understanding a Module

1. Read the module's README (e.g., [tensor.md](tensor.md))
2. Check the API reference in the document
3. Look at the source code for implementation details
4. Run the tests to see usage examples

### For Contributing to a Module

1. Read the existing documentation
2. Follow the patterns established in the code
3. Update documentation when adding new features
4. Add tests for new functionality

### For Debugging

1. Check the "Common Pitfalls" section of the relevant module
2. Use the debug output examples
3. Run with sanitizers enabled

---

## 🎓 Learning Path

### Path 1: Using MYTHOS.cpp for Inference

1. Read **[Tensor](tensor.md)** documentation
2. Read **[Model](model.md)** documentation
3. Read **[Tokenizer](tokenizer.md)** documentation
4. Read **[Usage Guide](../USAGE.md)**
5. Try the CLI tools

### Path 2: Training Models

1. Complete Path 1
2. Read **[Autograd](autograd.md)** documentation
3. Read **[Trainer](trainer.md)** documentation
4. Read **[Optimizer](optimizer.md)** documentation
5. Try training a small model

### Path 3: Understanding Quantization

1. Read **[Types](types.md)** documentation
2. Read **[Quantization](quantization.md)** documentation
3. Read **[OIL Format](oil_format.md)** documentation
4. Read **[Research Foundation](../RESEARCH.md)**
5. Experiment with different BPW settings

### Path 4: Contributing

1. Complete Path 1 and 2
2. Read **[Contributing Guide](../CONTRIBUTING.md)**
3. Read **[Architecture](../ARCHITECTURE.md)**
4. Read the source code
5. Start with a "good first issue"

---

## 📞 Need More Information?

- Check the **[API Reference](../API_REFERENCE.md)** for complete API documentation
- Check the **[Architecture](../ARCHITECTURE.md)** for system overview
- Check the **[Research Foundation](../RESEARCH.md)** for underlying science
- Check the source code - it's well-commented!
- Ask in GitHub Discussions or open an Issue

---

*Last updated: July 12, 2026*
