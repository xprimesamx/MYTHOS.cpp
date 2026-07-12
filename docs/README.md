# MYTHOS.cpp Documentation

> **M**ixed-format **Y**our-own **T**ensor **H**andcrafted **O**ptimized **S**ystem

**Zero-dependency C++20 AI Engine** - Train from scratch, fine-tune, quantize, and run inference, all within a single `.oil` binary format.

---

## 📚 Documentation Overview

Welcome to the comprehensive documentation for MYTHOS.cpp. This documentation is structured to help you understand, use, and contribute to the project.

### 🗂️ Documentation Structure

```
docs/
├── README.md                  # This file - Documentation index
├── ARCHITECTURE.md            # System architecture & design philosophy
├── BUILD.md                   # Build & installation guide
├── USAGE.md                   # Usage guide & examples
├── API_REFERENCE.md           # Complete API documentation
├── MODULES/                   # Per-module deep dives
│   ├── tensor.md
│   ├── autograd.md
│   ├── transformer.md
│   ├── quantization.md
│   ├── moe.md
│   ├── gpu_compute.md
│   └── ...
├── RESEARCH.md                # Research foundation & papers
├── CONTRIBUTING.md           # Contribution guidelines
├── INTERNAL/                  # Internal design documents
│   ├── memory_management.md
│   ├── kernel_optimization.md
│   └── format_design.md
└── EXAMPLES/                  # Code examples & tutorials
    ├── basic_inference.md
    ├── training_guide.md
    └── quantization_guide.md
```

---

## 🚀 Quick Start

If you're new to MYTHOS.cpp, start here:

1. **[BUILD.md](BUILD.md)** - Get the project compiled on your machine
2. **[USAGE.md](USAGE.md)** - Learn how to use the tools and APIs
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Understand how everything fits together

---

## 📖 Core Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | High-level system design, component relationships | Developers, Architects |
| [API_REFERENCE.md](API_REFERENCE.md) | Complete API documentation for all public interfaces | Developers, Users |
| [BUILD.md](BUILD.md) | Step-by-step build instructions for all platforms | Everyone |
| [USAGE.md](USAGE.md) | How to use MYTHOS.cpp for inference and training | Users |
| [RESEARCH.md](RESEARCH.md) | Research papers and algorithms that inspired the design | Researchers |

---

## 🏗️ Module Documentation

Detailed documentation for each major component:

| Module | Description | Files |
|--------|-------------|-------|
| **Tensor** | Multi-dimensional array with automatic differentiation support | `tensor.h`, `tensor.cpp` |
| **Autograd** | Automatic differentiation engine for training | `autograd.h`, `autograd.cpp` |
| **Math** | Mathematical operations with AVX2 optimization | `math.h`, `math.cpp`, `math_avx2.cpp` |
| **Transformer** | Transformer architecture implementation | `transformer.h`, `transformer.cpp` |
| **Quantization** | OIL4, OIL8, Ternary, Binary formats | `kernel_oil4.cpp`, `kernel_oil8.cpp`, `int8_quant.cpp`, etc. |
| **MoE** | Mixture of Experts implementation | `moe_variants.h`, `moe_variants.cpp` |
| **GPU Compute** | DirectX 12 GPU acceleration | `gpu_compute.h`, `gpu_compute.cpp` |
| **Model** | Model loading, saving, and management | `model.h`, `model.cpp` |
| **Tokenizer** | BPE tokenizer implementation | `bpe_tokenizer.h`, `bpe_tokenizer.cpp` |
| **Trainer** | Training loop and optimization | `trainer.h`, `trainer.cpp` |
| **OIL Format** | Binary format specification | `oil_format.h`, `oil_format.cpp` |

See **[MODULES/](MODULES/)** for detailed documentation on each component.

---

## 🔬 Research & Innovation

MYTHOS.cpp is built on a foundation of peer-reviewed research:

- **BitNet b1.58** - Ternary weights that match FP16 quality
- **AWQ** - Activation-aware weight quantization
- **VQ-VAE** - Vector quantization with learned codebooks
- **BitsMoE** - Per-expert bit-width allocation

See **[RESEARCH.md](RESEARCH.md)** for a complete list of research papers and how they're implemented.

---

## 🛠️ Development

Want to contribute or extend MYTHOS.cpp?

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guidelines for contributing
- **[INTERNAL/](INTERNAL/)** - Internal design documents
- **[EXAMPLES/](EXAMPLES/)** - Code examples and tutorials

---

## 📞 Need Help?

- Found a bug? Open an issue on GitHub
- Have a question? Check the documentation or open a discussion
- Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🎯 Project Status

| Component | Status | Tests | Documentation |
|-----------|--------|-------|---------------|
| Core Tensor | ✅ Stable | ✅ | ✅ |
| Autograd Engine | ✅ Stable | ✅ | ✅ |
| Transformer | ✅ Stable | ✅ | ✅ |
| OIL Format | ✅ Stable | ✅ | ✅ |
| Quantization Kernels | ✅ Stable | ✅ | ⏳ |
| MoE Variants | ✅ Stable | ✅ | ⏳ |
| GPU Compute | ✅ Experimental | ⚠️ | ⏳ |
| Training | ✅ Stable | ✅ | ⏳ |
| Inference | ✅ Stable | ✅ | ⏳ |

**Last Updated:** July 12, 2026
**Version:** v0.1

---

## 📄 License

MYTHOS.cpp is a proprietary project. See the main [README](../README.md) for details.


© 2026 MYTHOS.cpp Contributors
