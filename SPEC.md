# MYTHOS / OIL Engine — Mission Breakdown (Spec)

## 📦 Current State

### Protected Knowledge Dirs
| Dir | Files | Purpose |
|-----|-------|---------|
| `.bitnet` | 71 | Microsoft BitNet.cpp — full source (kernels: TL1/TL2/MAD/CUDA, codegen, converters) |
| `.claude` | 1 | VS Code settings only |
| `.git` | 179 | Git repo history |
| `.kilo` | 4 | Kilo CLI config |

### Empty Build-Time Folders
| Folder | Subfolders |
|--------|-----------|
| `TRAINER-ENGINE/` | `DENSE/`, `MoE/`, `MULTIMODEL/{AUDIO, EMBEDDINGS, IMAGE, OCR, TEXT, VIDEO}/` |
| `INFERENCE-ENGINE/` | — |
| `OIL8/` | — |

---

## 🧠 Knowledge Extracted from `.bitnet`

| Component | Tech | What It Does |
|-----------|------|-------------|
| `ggml-bitnet-mad.cpp` | AVX2/NEON | I2_S quant: weights → packed ternary (-1,0,+1), 2-bit storage, SIMD MAD compute |
| `ggml-bitnet-lut.cpp` | TL1 (ARM) / TL2 (x86) | LUT-based matmul: precomputed lookup tables for fast ternary × FP32 (no MAD) |
| `bitnet-kernels.cu` | CUDA | GPU kernels for ternary matmul |
| `codegen_tl1.py/tl2.py` | Python | Generates tuned TL1/TL2 kernel headers for specific model shapes |
| `gemm-config.h` | C macros | Block sizes (ROW_BLOCK_SIZE, COL_BLOCK_SIZE, PARALLEL_SIZE) per arch |
| `ggml-bitnet.h` | C API | `ggml_bitnet_mul_mat`, `transform_tensor`, `get_type_bits` |
| Converters | Python | HF/GGUF → BitNet format converters, embedding quantizers |

### Key Gap
BitNet.cpp is **inference-only** (wraps llama.cpp). No training, no fine-tune, no multi-format OIL8/OIL4.

---

## 🎯 Mission Parts (Breakdown)

### PART A: Format Layer — OIL8 / OIL4 / Mixed

| Sub-piece | Feasibility | Notes |
|-----------|-------------|-------|
| A1. OIL8 file spec (INT8 index + FP32 codebook) | ✅ Possible | Codebook = 256×FP32 per block; disk format = packed indices + codebook |
| A2. OIL4 file spec (INT4 index + FP16 codebook) | ✅ Possible | Same structure, 16 centroids |
| A3. Mixed format header (OIL8/OIL4/OIL2 per layer) | ✅ Possible | Per-layer type field in file header |
| A4. Integer/decimal/rational exact storage | ⚠️ Partial | Exact storage needs variable codebook or residual. Pure VQ loses some values. **100% exact requires per-value storage → size grows** |
| A5. 75% disk reduction vs FP32 (OIL8) | ✅ Possible | 4B → 1B index + ~1KB codebook = ~4× smaller |
| A6. 0% quality loss guarantee | ⚠️ Misleading | Impossible **always**. Achievable: train-into-format, or VQ + residual. "Practical zero" (perplexity diff <0.01) = doable with fine-tune |

### PART B: TRAINER-ENGINE (Training)

| Sub-piece | Feasibility | Notes |
|-----------|-------------|-------|
| B1. Pure C++ tensor library | ✅ Possible | Huge effort. Alternatives: eigen, xtensor (but no AI deps). **Must build custom** |
| B2. Dense transformer train | ✅ Possible | Attention, FFN, LayerNorm, AdamW — well-known, CUDA-free CPU path works for small models |
| B3. MoE train | ✅ Possible | Router + experts + load balancing — more complex but proven |
| B4. Multimodal train (text/image/video/audio/embed/OCR) | ✅ Possible (phased) | Each modality = different encoders, data pipelines. **Phase 1: text. Phase N: rest** |
| B5. OIL-native training (train directly in compressed space) | ✅ Possible | VQ training with codebook update (k-means + centroid gradient). Research-backed |
| B6. LoRA/QLoRA-style fine-tune | ✅ Possible | All math: inject low-rank adapters, quant base, train adapters |
| B7. 48T+ scale design | ✅ Possible for engine | Distributed data/model parallelism, sharding protocols, FSDP-like |
| B8. 48T train on single PC | ❌ Impossible | Even OIL compressed = terabytes. Hardware physics |
| B9. ~5-10% less compute vs PyTorch | ✅ Possible | C++ overhead less than Python; fused ops; no Python-GIL. But PyTorch also has C++ backend. Real win: fused custom kernels |
| B10. Train on this PC (~14GB, iGPU) | ✅ Limited | 0.1B-0.4B full train; 1B-3B LoRA fine-tune. **Small models only** |

### PART C: INFERENCE-ENGINE

| Sub-piece | Feasibility | Notes |
|-----------|-------------|-------|
| C1. Load OIL8/OIL4 file format | ✅ Possible | Custom loader/serializer |
| C2. CPU kernels for OIL matmul | ✅ Possible | Lookup + MAD from `.bitnet` knowledge |
| C3. Auto-regressive generation | ✅ Possible | KV cache, top-k/top-p, sampling |
| C4. 512+ tok/s any hardware + any model | ❌ Impossible | Physics: memory bandwidth + flops. 7B on CPU = ~10-40 tok/s. 512+ possible on strong GPU + small model |
| C5. Chat interface | ✅ Possible | stdin/stdout or simple server |

### PART D: System / Infrastructure

| Sub-piece | Feasibility | Notes |
|-----------|-------------|-------|
| D1. CMake build system | ✅ Possible | Already have CMakeLists.txt pattern from `.bitnet` |
| D2. Zero Python/AI deps | ✅ Possible | All C++. Just need standard lib, optionally OpenMP/pthreads |
| D3. Custom kernel generation | ✅ Possible | Pattern from `.bitnet/codegen_tl1.py/tl2.py` — Python codegen for C++ headers |
| D4. Cross-platform | ✅ Possible | Windows, Linux, macOS (with SIMD paths for each) |
| D5. VS2022 + Clang build on Windows | ✅ Possible | BitNet already does it |

### PART E: Competitive Differentiation

| vs llama.cpp | vs BitNet.cpp | OIL Engine Advantage |
|-------------|---------------|---------------------|
| Only inference | Only inference | **Train + Infer** |
| GGUF format | Only ternary | **OIL8/OIL4/OIL2 mix** |
| Python for train | Python for setup | **Pure C++ end-to-end** |
| FP16/8-bit quant | 1.58-bit only | **Multiple bit-widths per layer** |
| No native fine-tune | No fine-tune | **LoRA/QLoRA built-in** |

---

## 🗺️ Execution Order (Recommended)

```
Phase 1: Foundation
  ├── OIL8 file format spec + loader/saver (C++)
  ├── OIL4 file format spec + loader/saver (C++)
  ├── Math primitives (tensor, vec, mat, basic ops)
  └── CMake project structure

Phase 2: INFERENCE-ENGINE v1
  ├── OIL8 CPU matmul kernel (lookup-based)
  ├── Basic transformer forward pass (pre-built weights)
  ├── KV cache + sampling
  └── Chat loop (CLI)

Phase 3: TRAINER-ENGINE v1 — Text Only
  ├── AdamW optimizer
  ├── Dense transformer backward pass
  ├── Forward/backward on small models (demo scale)
  └── Training loop + checkpoint

Phase 4: OIL-Native Training
  ├── VQ training (k-means codebook init + centroid update)
  ├── Straight-Through Estimator for low-bit
  ├── Quantization-aware fine-tune
  └── LoRA adapter training

Phase 5: Scale & Performance
  ├── MoE training support
  ├── Distributed training hooks (48T design)
  ├── Kernel optimization (SIMD tuning, codegen)
  └── Benchmark suite (vs baseline, vs llama.cpp, vs BitNet)

Phase 6: Multimodal Expansion
  ├── Image encoder/decoder
  ├── Audio encoder/decoder
  ├── Video support
  ├── Embedding models
  └── OCR module

Phase 7: Production Readiness
  ├── Memory optimization
  ├── Cross-platform testing
  ├── Documentation
  └── API/server interface
```

---

## ⚠️ Honest Flags (Do NOT Overpromise)

| Statement | Verdict |
|-----------|---------|
| "100% lossless always" | ❌ — Info theory: compress → information loss. **Practical near-lossless = yes** |
| "512+ tok/s on any hardware" | ❌ — Weak HW + large model → single digits |
| "48T train on 14GB RAM" | ❌ — Impossible regardless of format |
| "Better than GPT-4 at 100× smaller" | ❌ — Scaling laws are real |
| "Rivals llama.cpp first day" | ❌ — They have years of community optimization |
| "Zero code reuse from BitNet" | ❌ — Studying their kernels is the whole point of `.bitnet` |
| "All 7 phases done in 1 week" | ❌ — Years-long project for solo/team |

---

## ✅ What IS 100% Provably Achievable

- **Working C++ engine** that loads OIL8 files and runs inference
- **Train small models (0.1B-0.4B)** entirely in C++
- **Fine-tune 1B-3B models** with LoRA-style adapters
- **Disk reduction ~4× vs FP32** for OIL8 format
- **OIL8 quality near FP32** with proper VQ + fine-tune
- **Clean separation** of TRAINER and INFERENCE engines
- **Multi-format per-layer** (OIL8 for sensitive, OIL4/OIL2 for tolerant)
- **Phase-by-phase delivery** — each phase independently useful
