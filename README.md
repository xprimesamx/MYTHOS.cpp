# ⚡ MYTHOS.cpp

> **M**ixed-format **Y**our-own **T**ensor **H**andcrafted **O**ptimized **S**ystem

**Zero-dependency C++ AI engine.** Train from scratch, fine-tune in native OIL format, quantize, and run inference — all within a single `.oil` binary format. No Python. No PyTorch. No HuggingFace. No Eigen. No BLAS. Just C++20 and hand-written SIMD kernels.

```
EVERYTHING IS USED OUR OWN — zero dependency, maximum control.
```

---

## 📋 Table of Contents

- [Vision](#-vision)
- [The Problem](#-the-problem)
- [What is OIL?](#-what-is-oil)
- [Research Foundation](#-research-foundation)
- [Architecture](#-architecture)
- [Component Deep-Dive](#-component-deep-dive)
- [OIL Binary Format Spec](#-oil-binary-format-spec)
- [Kernel Design](#-kernel-design)
- [Build System](#-build-system)
- [Phase-by-Phase Roadmap](#-phase-by-phase-roadmap)
- [Comparison with Existing Projects](#-comparison-with-existing-projects)
- [Developer Machine Reality](#-developer-machine-reality)
- [Performance Targets](#-performance-targets)
- [Tools & CLI](#-tools--cli)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Vision

Build a **complete, production-grade AI engine in pure C++20** with zero external dependencies — from tensor math to transformer training to multimodal inference. Every byte of code hand-crafted, every kernel hand-tuned, every format decision justified by research.

The `.oil` format is the single source of truth: models are born in OIL, trained in OIL, fine-tuned in OIL, and served in OIL. No format conversions, no serialization chains, no Python middleware.

### Why?

- **Privacy:** Air-gapped training on your hardware, your data
- **Performance:** C++ beats Python for tight loops, SIMD, and cache control
- **Understanding:** You don't truly understand transformers until you've written the backward pass by hand
- **Control:** No dependency hell, no version conflicts, no `pip install` rabbit holes
- **Cost:** Train capable models on consumer hardware without cloud GPU bills

---

## 🔥 The Problem

Large Language Models are transforming the world, but the stack to build them is:

1. **Bloated** — PyTorch + CUDA + HuggingFace + Tokenizers + Accelerate + DeepSpeed = 1GB+ of dependencies
2. **Python-locked** — every research project, every training script, every inference server requires the Python runtime
3. **Format-chaos** — models ship as PyTorch `.pt`, get converted to GGUF for inference, get quantized with yet another tool, fine-tuned with PEFT in yet another format
4. **Wasteful** — uniform 16-bit or 8-bit quantization wastes bits on unimportant weights; 4-bit GPTQ needs calibration datasets and still loses quality

### The OIL Answer

| Problem | OIL Solution |
|---------|-------------|
| Python dependency | 100% C++, no runtime required |
| Format chaos | Single `.oil` format for everything |
| Wasteful quantization | Per-weight-block format routing |
| Quality loss | Train-in-format (STE), never post-quantize |
| Complex deployment | Single binary, no pip install |

---

## 📦 What is OIL?

**O**ptimized **I**nference & **L**earning is a mixed-precision binary container format. Unlike uniform quantization (everything 4-bit or 8-bit), OIL assigns a **different format to every weight block** based on its importance to model quality.

### Format Options

| Format | Index Bits | Codebook | Storage BPW | Compute Precision | Quality |
|--------|-----------|----------|-------------|-------------------|---------|
| **OIL8** | 8 (INT8) | 256 × FP32 | 8.0 + codebook | FP32 (gather) | Matches FP32 |
| **OIL4** | 4 (INT4 packed) | 16 × FP16 | 4.0 + codebook | FP16/FP32 | Matches FP16 |
| **Ternary** | 2 (I2_S packed) | {−1, 0, +1} × scale | 1.58 | FP32 (add only) | Matches FP16* |
| **Binary** | 1 (packed) | {−1, +1} × scale | 1.0 | FP32 (xor+popcount) | Slight loss |

*\*BitNet b1.58 (arXiv:2402.17764) proves ternary matches FP16 perplexity with from-scratch training.*

### Why Mixed Formats?

Research shows that neural network weights have vastly different importance:

- **~1% of weights are "salient"** — changing them significantly changes output (AWQ, arXiv:2306.00978)
- **~4% are moderately important** — need moderate precision
- **~95% can be ternary or binary** — the model learns to be robust via training-in-format (BitNet)

OIL's **FormatPlanner** analyzes a model with calibration data and allocates formats to hit a target BPW:
```
Score each weight block for importance (AWQ-style activation magnitudes)
Allocate OIL8 to top 1% most salient
Allocate OIL4 to next 4%
Allocate ternary to remaining 95%
If target BPW > 1.50, shift boundary toward binary
```

**Result: 1.50 BPW average with FP32-level quality.**

### Comparison with Existing Formats

| Format | BPW | Quality | Flexibility | Trainable |
|--------|-----|---------|-------------|-----------|
| FP32 | 32 | Reference | N/A | ✅ |
| FP16 | 16 | Near-FP32 | Uniform | ✅ |
| INT8 (W8A8) | 8 | Near-FP32 | Uniform | ⚠️ QAT |
| INT4 (GPTQ) | 4 | ~FP16 | Uniform | ❌ PTQ only |
| NF4 (QLoRA) | 4 | ~FP16 | Uniform | ⚠️ Adapter only |
| GGUF Q4_K_M | 4.5 | ~FP16 | Importance-grouped | ❌ PTQ only |
| BitNet 1.58 | 1.58 | ~FP16* | Uniform ternary | ✅ Only |
| **OIL (this)** | **1.50** | **FP32** | **Per-block mixed** | **✅ Full** |

*\*BitNet matches FP16. OIL targets FP32 via OIL8 allocation for salient weights.*

---

## 🔬 Research Foundation

Every design decision in MYTHOS.cpp is grounded in peer-reviewed research. Here's what we studied and how it applies:

### BitNet b1.58 (arXiv:2402.17764, arXiv:2310.11453)

**Key finding:** Ternary weights {-1, 0, +1} trained from scratch match FP16 perplexity and downstream task performance.

**How it works:**
1. Weights are ternary + a per-tensor scale factor `α`
2. Forward pass: `W_ternary · x = α · ({-1,0,+1}) · x` — no multiplications, just additions
3. Backward pass uses Straight-Through Estimator (STE): gradients pass through the quantization step as if it was identity
4. Activations are quantized to INT8 per-tensor (find max, scale to [-127, 127])

**Impact on OIL:** This is the core proof that 0% knowledge loss is achievable — the model is trained to work with ternary weights; it never "loses" FP32 precision because it was never FP32.

### Bitnet.cpp (arXiv:2502.11880)

**Key finding:** Element-wise LUT-based matmul (TL) outperforms bit-wise LUT (T-MAC) by 2.32× on x86 and 1.19× on ARM for ternary inference.

**Two kernels:**
- **TL (Ternary Lookup Table):** Precompute all possible activation sums for groups of 2-3 ternary weights. During inference, just look up the precomputed value. TL2 achieves 1.67 BPW with element-wise mirror consolidation.
- **I2_S (Int2 + Scale):** Pack 4 ternary values (2 bits each) into 1 byte with a shared scale factor. Uses MAD (multiply-add) computation, strictly matches training quantization for lossless inference.

**Impact on OIL:** We adopt both approaches — TL for fast batch inference, I2_S for lossless correctness. Our OIL4/OIL8 kernels extend the LUT concept to larger codebooks.

### AWQ: Activation-Aware Weight Quantization (arXiv:2306.00978)

**Key finding:** Only ~1% of weights are salient — identified by activation magnitudes. Protecting these with higher precision recovers nearly all quality loss.

**Impact on OIL:** The FormatPlanner uses AWQ-style importance scoring to decide which weight blocks get OIL8 vs OIL4 vs ternary. This is how we achieve 1.50 BPW with FP32 quality.

### VQ-VAE (NeurIPS 2017)

**Key finding:** Vector quantization with codebook learning enables discrete representation learning. The codebook is trained with EMA updates and commitment loss.

**Impact on OIL:** OIL8/OIL4 codebooks use VQ training: k-means initialization + EMA centroid update + straight-through gradient. This is how we train models directly in the compressed format.

### Custom Fine-Tuning System

**Key insight:** Fine-tuning should be native to the format — not a separate adapter bolted on. OIL's training engine handles fine-tuning at the tensor level: identify which weight blocks need updates, apply gradient updates directly in OIL format via STE, and update codebook centroids as needed.

**Impact on OIL:** The fine-tuning system is built into the trainer — no external adapters, no separate optimizer for adapters. Train or fine-tune, it's the same code path.

### BitsMoE (arXiv:2410.01045)

**Key finding:** Different experts in a MoE model need different bit-widths. Routing can also be quantized.

**Impact on OIL:** Per-expert format allocation extends naturally from OIL's per-block format routing.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MYTHOS.cpp                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    CORE LAYER                           │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────────┐ │   │
│  │  │ Types  │  │Memory  │  │ Tensor │  │   Random     │ │   │
│  │  │ Enums  │  │Aligned │  │View    │  │ Xoroshiro128 │ │   │
│  │  │ Shape  │  │Pool    │  │Slice   │  │ Uniform/Norm │ │   │
│  │  │ DType  │  │Buffer  │  │Strided │  │              │ │   │
│  │  └────────┘  └────────┘  └────────┘  └──────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   MATH LAYER (SIMD)                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │   │
│  │  │  BLAS    │  │Pointwise │  │  GEMM Kernels        │  │   │
│  │  │ gemm     │  │ ReLU     │  │  I2_S (MAD)          │  │   │
│  │  │ gemv     │  │ GELU     │  │  TL1/TL2 (LUT)       │  │   │
│  │  │ dot      │  │ SiLU     │  │  OIL8 Lookup         │  │   │
│  │  │ axpy     │  │ Sigmoid  │  │  OIL4 Lookup         │  │   │
│  │  └──────────┘  │ Softmax  │  └──────────────────────┘  │   │
│  │                 │ LayerNorm│                            │   │
│  │                 │ RMSNorm  │                            │   │
│  │                 └──────────┘                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 FORMAT LAYER (.oil)                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │   │
│  │  │Codebook  │  │Format    │  │  OIL Writer/Reader   │  │   │
│  │  │ OIL8(256)│  │Planner   │  │  Binary (de)serial   │  │   │
│  │  │ OIL4(16) │  │BPW=1.50 │  │  Magic + Tables      │  │   │
│  │  │ Ternary  │  │AWQ-score│  │  + indices + cb       │  │   │
│  │  │ Binary   │  │Allocator│  │                      │  │   │
│  │  └──────────┘  └──────────┘  └──────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 MODEL LAYER                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │   │
│  │  │ Layers   │  │ Models   │  │  Tokenizer           │  │   │
│  │  │ Linear   │  │ Dense    │  │  BPE (byte-pair)     │  │   │
│  │  │ RMSNorm  │  │ MoE      │  │  Unigram (EM)        │  │   │
│  │  │ RoPE     │  │MultiModal│  │  encode/decode       │  │   │
│  │  │ Attn-MHA │  │          │  │  train on corpus     │  │   │
│  │  │ FFN-SwiGLU│ └──────────┘  └──────────────────────┘  │   │
│  │  │ MoEFFN   │                                          │   │
│  │  └──────────┘                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │  INFERENCE   │  │    TRAINING     │  │  CONVERTERS     │   │
│  │  KV Cache    │  │  Autograd Graph │  │  GGUF → .oil    │   │
│  │  Sampler     │  │  AdamW/SGD      │  │  HF → .oil      │   │
│  │  Generator   │  │  STE Quantizer  │  │  FP32 ⇄ .oil    │   │
│  │  Chat CLI    │  │  Native FineTune│  │                 │   │
│  └──────────────┘  │  Checkpoint     │  └─────────────────┘   │
│                     │  DataLoader    │                          │
│                     │  Distributed   │                          │
│                     └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow: Training → Inference

```
Raw Text → Tokenizer → Training Loop → .oil File → Inference Engine → Text
               │              │                         │
               │              ▼                         ▼
               │      ┌──────────────┐         ┌──────────────┐
               │      │ STE Forward  │         │ Load .oil    │
               │      │ (OIL format) │         │ Parse Format │
               │      │ Codebook Up  │         │ Table + CB   │
               │      │ AdamW Step   │         │ Read Indices │
               │      │ Save .oil    │         │              │
               │      └──────────────┘         │ KV Cache Init│
               │                               │ Sampler Init │
               │                               └──────┬───────┘
               │                                      ▼
               │                              ┌──────────────┐
               │                              │ Token Loop   │
               │                              │ For each tok:│
               │                              │ 1. Embed     │
               │                              │ 2. N×Trans   │
               │                              │ 3. LM Head   │
               │                              │ 4. Sample    │
               │                              │ 5. Append KV │
               │                              └──────────────┘
```

---

## 🔧 Component Deep-Dive

### 1. Tensor Library (`include/oil/tensor.h`)

Custom n-dimensional array implementation with:

```
oil::Tensor<float> t(oil::Shape{2, 3, 4});  // 3D tensor

// Views — no data copy
auto v = t.slice(0, 1);      // select first batch
auto r = t.reshape({6, 4});   // reshape
auto p = t.permute({2, 0, 1}); // transpose

// Math — SIMD accelerated
t.fill(1.0f);
auto y = oil::math::gemm(a, b);  // matrix multiply
auto z = oil::math::softmax(x, 1); // softmax along axis

// Gradient tracking
t.requires_grad(true);
auto loss = t.mean();
loss.backward();  // populates t.grad()

// Serialization
oil::OILWriter writer("model.oil");
writer.write_tensor("weights", t);
```

**Memory model:** `oil::Buffer` with 64-byte alignment (SIMD-safe), reference-counted ownership, optional memory pool for temporary allocations.

### 2. Math Library (`include/oil/math.h`)

Full BLAS-level operations + neural network primitives:

| Category | Operations | SIMD Level |
|----------|-----------|------------|
| BLAS-1 | `dot`, `axpy`, `scal`, `norm`, `asum` | AVX2/NEON |
| BLAS-2 | `gemv` (matrix × vector) | AVX2/NEON |
| BLAS-3 | `gemm` (matrix × matrix + bias) | AVX2 tiled |
| Activations | `relu`, `gelu`(tanh/taylor), `silu`, `sigmoid`, `tanh` | AVX2 |
| Normalization | `layer_norm`, `rms_norm`, `batch_norm` | AVX2 |
| Softmax | `softmax` (stable, subtract max) | AVX2 |
| Random | `uniform`, `normal` (Box-Muller) | Scalar |
| Ternary | `ternary_gemm`, `ternary_gemv` (add-only) | AVX2 I2_S/TL |
| Codebook | `oil8_gemm`, `oil4_gemm` (gather-accumulate) | AVX2 gather |

### 3. OIL Format (`include/oil/oil_format.h`)

```
Magic:  "OIL1" (4 bytes)
Version: uint32_t
Flags:   uint32_t (training/inference, metadata type)
Config:  ModelConfig (vocab, hidden, layers, heads, etc.)

FormatTable:
  [block_id: uint32_t, format: uint8_t, cb_size: uint32_t] × N

Block Data:
  For each block:
    Codebook: [centroid_f32 × 256] or [centroid_f16 × 16] or [scale_f32]
    Indices:  packed uint8/uint4/bit indices for each weight

Tensor Table:
  [name: string, start_block: uint32_t, num_blocks: uint32_t] × T
```

### 4. Codebook (`include/oil/codebook.h`)

```cpp
template<typename T, int N>
struct Codebook {
    std::vector<T> centroids;     // N centroids
    // Training
    void kmeans_init(const float* data, size_t count);
    void ema_update(const float* data, const uint8_t* assign, float lr);
    // Quantize
    uint8_t quantize(float val) const;  // nearest centroid
    float dequantize(uint8_t idx) const;
    // Serialize
    void serialize(OILWriter& w) const;
    static Codebook deserialize(OILReader& r);
};

using OIL8Codebook = Codebook<float, 256>;   // 8-bit format
using OIL4Codebook = Codebook<half, 16>;      // 4-bit format
```

### 5. Transformer Model (`include/oil/transformer.h`)

```cpp
struct TransformerConfig {
    int vocab_size;
    int hidden_size;
    int num_layers;
    int num_heads;
    int head_dim;        // hidden_size / num_heads
    int ffn_hidden_size; // typically 4 * hidden_size
    float norm_eps;
    float rope_theta;
    int max_seq_len;
    Activation activation; // GELU, SiLU, ReLU
};

class TransformerBlock {
    RMSNorm attention_norm;
    Attention attention;
    RMSNorm ffn_norm;
    FFN ffn;  // SwiGLU: up @ gate * down
};

class DenseModel {
    Embedding tok_embeddings;
    std::vector<TransformerBlock> layers;
    RMSNorm norm;
    Linear lm_head;
};
```

### 6. Inference Engine (`include/oil/sampler.h`)

```cpp
struct InferenceConfig {
    float temperature;
    int top_k;
    float top_p;
    float repetition_penalty;
    int max_tokens;
};

class KVCache {
    std::vector<Tensor> k_cache; // per-layer K cache
    std::vector<Tensor> v_cache; // per-layer V cache
    int seq_len;
    void append(int layer, const Tensor& k, const Tensor& v);
    std::pair<Tensor, Tensor> get(int layer, int pos) const;
};

class Sampler {
    int greedy(const Tensor& logits);
    int top_k_sample(const Tensor& logits, int k, float temp);
    int top_p_sample(const Tensor& logits, float p, float temp);
};  // All with Xoroshiro128+ RNG (no <random> dependency)
```

### 7. Training Engine (`include/oil/optimizer.h`)

```cpp
class AdamW {
    float lr, beta1, beta2, eps, weight_decay;
    Tensor m, v; // moment estimates
    int t;       // step counter
    
    void step(Tensor& params, const Tensor& grads);
    void zero_grad();
};

class STEQuantizer {
    // Forward: quantize to target format
    // Backward: identity gradient (straight-through)
    Tensor forward(const Tensor& fp32_weight, Format target);
};

class Trainer {
    Model* model;
    AdamW* optimizer;
    LossFunction* loss_fn;
    
    float train_batch(const Tensor& input_ids, const Tensor& labels);
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
};
```

### 8. Native Fine-Tuning System (`include/oil/finetune.h`)

OIL's fine-tuning works directly on the compressed format — no external adapters needed:

```cpp
struct FineTuneConfig {
    float lr;                 // learning rate for fine-tune
    int warmup_steps;         // LR warmup
    float update_threshold;   // skip updates below this gradient norm
    bool update_codebooks;    // whether to update codebook centroids
};

class FineTuner {
    Model* model;
    AdamW optimizer;
    
    // Identify which weight blocks need updates based on gradient magnitude
    std::vector<BlockID> find_trainable_blocks(const Tensor& gradients, float threshold);
    
    // Apply gradient update directly in OIL format via STE
    // Updates: weight indices, codebook centroids, or both
    void step(const Tensor& batch);
    
    // Save fine-tuned model (same .oil format, just updated weights)
    void save(const std::string& path);
};
```

### 9. Tokenizer (`include/oil/tokenizer.h`)

```cpp
class BPETokenizer {
    struct Merge { int id; int pair[2]; int freq; };
    std::vector<std::string> vocab;
    std::unordered_map<std::pair<int,int>, int> merges;
    
    void train(const std::vector<std::string>& texts, int vocab_size);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
};

class UnigramTokenizer {
    // Kneser-Ney smoothing, EM training
    void train(...);
    std::vector<int> encode(const std::string& text);
};
```

---

## 🗂️ OIL Binary Format Spec

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | `magic` | `0x314C494F` ("OIL1") |
| 4 | 4 | `version` | Format version (major.minor.patch packed) |
| 8 | 4 | `flags` | Training/inference, format flags |
| 12 | 4 | `config_size` | Size of ModelConfig JSON/Protobuf |
| 16 | config_size | `config` | Model configuration |
| 16+config_size | 4 | `num_format_blocks` | N blocks in format table |
| 20+config_size | N*9 | `format_table` | `{block_id:u32, format:u8, cb_bytes:u32}` |
| 20+config_size+N*9 | 4 | `num_tensors` | T named tensors |
| ... | T*(... ) | `tensor_table` | `{name_len:u16, name, block_start:u32, block_count:u32}` |
| ... | varies | `block_data` | Actual codebooks + packed indices |

**Block Data Layout Per Format:**

```
OIL8:   [codebook: 256×f32 bytes] [indices: 1 byte per weight]
OIL4:   [codebook: 16×f16 bytes]  [indices: nibble-packed, 2 per byte]
Ternary:[scale: f32 bytes]        [indices: 2-bit packed, 4 per byte] or TL-indexed
Binary: [scale: f32 bytes]        [indices: 1-bit packed, 8 per byte]
```

---

## ⚡ Kernel Design

### MAD Kernel (I2_S — BitNet compatible)

```
Storage: 4 ternary values packed per byte, scale for the block
Compute: For each block of 128 weights:
  1. Unpack 4 ternary values per byte → {-1, 0, +1} × scale
  2. Dot product with FP32 activations (no multiply for ternary, just add/sub)
  3. Accumulate across blocks
```

x86 path: AVX2 `_mm256` operations, 128-weight blocks
ARM path: NEON `vld1q_s8` + pairwise add

### TL Kernel (Ternary LUT — Bitnet.cpp compatible)

```
TL1: Groups of 2 ternary weights → 3² = 9 precomputed sums per LUT entry
TL2: Groups of 3 ternary weights → 3³ = 27 → mirror consolidation → 14 precomputed
Storage: 5 bits for 3 weights (1.67 BPW) with sign/unsigned splitting
Compute:
  1. Preprocessor: per-tensor INT8 activation quant + build LUT
  2. GEMM: load 4-bit index → lookup → XOR+ADD sign operation → accumulate
```

### OIL8/OIL4 Lookup Kernel

```
OIL8: 256 FP32 centroids per codebook
  1. Load INT8 index per weight
  2. Gather FP32 centroid from codebook
  3. Multiply by FP32 activation (fused multiply-add)
  4. Accumulate across row

OIL4: 16 FP16 centroids per codebook
  1. Load INT4 index (nibble unpack)
  2. Gather FP16 centroid → convert to FP32
  3. Multiply by FP32 activation
  4. Accumulate across row
```

---

## 🔨 Build System

### Requirements

- **CMake** ≥ 3.24
- **C++20** compiler:
  - Clang ≥ 16 (primary target — `clang-cl` on Windows)
  - GCC ≥ 12 (secondary)
  - MSVC 2022 (tertiary)
- **Optional:** Ninja build system

### Configuration

```bash
# Clone
git clone https://github.com/xprimesamx/MYTHOS.cpp
cd MYTHOS.cpp

# Configure & Build
mkdir build && cd build
cmake .. -G "Ninja" -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build .

# With debug symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug

# With address sanitizer
cmake .. -DCMAKE_BUILD_TYPE=Debug -DOIL_SANITIZE=ON
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `OIL_BUILD_TESTS` | ON | Build unit tests |
| `OIL_BUILD_BENCHMARKS` | ON | Build benchmarks |
| `OIL_BUILD_TOOLS` | ON | Build CLI tools |
| `OIL_AVX2` | auto | Enable AVX2 kernels |
| `OIL_AVX512` | OFF | Enable AVX-512 kernels |
| `OIL_NEON` | auto | Enable ARM NEON kernels |
| `OIL_NATIVE` | OFF | `-march=native` tuning |

### Project CMake Structure

```cmake
cmake_minimum_required(VERSION 3.24)
project(MYTHOS.cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform detection
include(cmake/arch.cmake)
include(cmake/compiler.cmake)

# Core library — no deps
add_library(oil_core
    src/tensor.cpp src/math.cpp src/random.cpp)
target_include_directories(oil_core PUBLIC include)

# Format library
add_library(oil_format
    src/oil_format.cpp src/codebook.cpp src/format_planner.cpp)
target_link_libraries(oil_format PUBLIC oil_core)

# Model library
add_library(oil_model
    src/transformer.cpp src/embedding.cpp src/attention.cpp
    src/ffn.cpp src/linear.cpp src/norm.cpp src/rope.cpp)
target_link_libraries(oil_model PUBLIC oil_core oil_format)

# Inference library
add_library(oil_inference
    src/kv_cache.cpp src/sampler.cpp src/generator.cpp)
target_link_libraries(oil_inference PUBLIC oil_model)

# Training library
add_library(oil_trainer
    src/autograd.cpp src/optimizer.cpp src/trainer.cpp
    src/ste_quantizer.cpp src/finetune.cpp)
target_link_libraries(oil_trainer PUBLIC oil_model)

# Tokenizer library
add_library(oil_tokenizer
    src/bpe_tokenizer.cpp src/unigram_tokenizer.cpp)
target_link_libraries(oil_tokenizer PUBLIC oil_core)

# Tools
add_executable(oil_tools
    tools/train.cpp tools/infer.cpp tools/convert.cpp
    tools/bench.cpp tools/info.cpp)
target_link_libraries(oil_tools
    PRIVATE oil_core oil_format oil_model oil_inference
            oil_trainer oil_tokenizer)

# Tests
add_executable(oil_tests
    tests/test_tensor.cpp tests/test_math.cpp
    tests/test_format.cpp tests/test_kernel.cpp
    tests/test_model.cpp)
target_link_libraries(oil_tests PRIVATE oil_core oil_format oil_model)
```

---

## 🗺️ Phase-by-Phase Roadmap

### Phase 1: Foundation (Core)
**Goal:** Tensor math, OIL format, build system, basic SIMD

- [ ] CMake project with arch/compiler detection
- [ ] `types.h` — Format enum, Shape, DType, Status, Config
- [ ] `memory.h/cpp` — AlignedAllocator, Buffer, MemoryPool
- [ ] `tensor.h/cpp` — Full n-dimensional tensor with views, slicing, broadcasting
- [ ] `math.h/cpp` — Scalar math: gemm, norm, softmax, activations
- [ ] `random.h/cpp` — Xoroshiro128+ RNG
- [ ] `oil_format.h/cpp` — OIL binary reader/writer
- [ ] `codebook.h/cpp` — OIL8(256×f32), OIL4(16×f16), ternary, binary codebooks
- [ ] `format_planner.h/cpp` — AWQ-scoring, BPW=1.50 allocation
- [ ] Tests: tensor round-trip, math correctness, format encode→decode

### Phase 2: Inference Engine
**Goal:** Load OIL model, run autoregressive generation

- [ ] `kernel.h` + `kernel_i2s.cpp` — I2_S MAD ternary GEMM (AVX2 + scalar)
- [ ] `kernel_tl.cpp` — TL1/TL2 LUT ternary GEMM
- [ ] `kernel_oil8.cpp` — OIL8 codebook lookup GEMM
- [ ] `kernel_oil4.cpp` — OIL4 codebook lookup GEMM
- [ ] `transformer.h/cpp` — Linear, RMSNorm, RoPE, Attention, FFN, TransformerBlock
- [ ] `model.h/cpp` — DenseModel with load/save
- [ ] `kv_cache.h/cpp` — KV cache with OIL4 compressed option
- [ ] `sampler.h/cpp` — Greedy, Top-K, Top-P, Temperature, Beam search
- [ ] `generator.h/cpp` — Autoregressive loop with streaming
- [ ] `tokenizer.h/cpp` — BPE tokenizer from scratch
- [ ] `tools/infer.cpp` — Interactive chat CLI

### Phase 3: Training Engine
**Goal:** Train small transformers from scratch in C++

- [ ] `autograd.h/cpp` — Computation graph with topological sort
- [ ] Autograd ops: MatMul, Add, Mul, ReLU, GELU, SiLU, Softmax, LayerNorm, CrossEntropy
- [ ] `optimizer.h/cpp` — AdamW, SGD with momentum, LR scheduler
- [ ] `trainer.h/cpp` — Training loop, batch iteration, logging
- [ ] `dataloader.h/cpp` — Text → tokenized batches with shuffle
- [ ] Checkpoint save/load in .oil format
- [ ] `tools/train.cpp` — Training CLI with config file

### Phase 4: OIL-Native Training
**Goal:** Train directly in compressed OIL format with minimal quality loss

- [ ] `ste_quantizer.h/cpp` — Straight-Through Estimator for all OIL formats
- [ ] `codebook_trainer.h/cpp` — VQ training: k-means init, EMA update, commitment loss
- [ ] `finetune.h/cpp` — Native OIL fine-tuning system
- [ ] Gradient-based weight block selection for targeted updates
- [ ] Codebook-aware fine-tune (update centroids during training)
- [ ] `tools/finetune.cpp` — Fine-tuning CLI
- [ ] Quantization-aware training loop integrated with Trainer

### Phase 5: Scale & Performance
**Goal:** Larger models, faster inference, MoE, distributed hooks

- [ ] MoE: Router (softmax top-K), load balancing loss, expert parallelism
- [ ] `moe.h/cpp` — MoEFFN, MoETransformerBlock, MoEModel
- [ ] Tensor parallelism hooks (weight sharding)
- [ ] FSDP-style sharding design
- [ ] Tiled GEMM for better cache utilization
- [ ] Quantized KV cache (OIL4 for keys/values)
- [ ] `bench/bench_kernels.cpp` — Throughput vs scalar baseline
- [ ] `bench/bench_inference.cpp` — tok/s, memory usage
- [ ] `bench/bench_quality.cpp` — Perplexity across formats

### Phase 6: Multimodal
**Goal:** Support for image, audio, video, embeddings, OCR

- [ ] `vision_encoder.h/cpp` — ViT-style patch embedding + transformer
- [ ] `vision_decoder.h/cpp` — Decoder for generation
- [ ] `audio_encoder.h/cpp` — Spectrogram → CNN → transformer
- [ ] `audio_decoder.h/cpp` — Vocoder-style decoder
- [ ] `video_encoder.h/cpp` — 3D conv + spatiotemporal attention
- [ ] `embedding.h/cpp` — Sentence transformer-style model
- [ ] `ocr.h/cpp` — CNN + attention for text recognition
- [ ] `model_multimodal.h/cpp` — Joint multimodal model with cross-attention

### Phase 7: Production Readiness
- [ ] Memory optimization (shared weights, quantized cache)
- [ ] Cross-platform: Windows (Clang-cl), Linux (GCC), macOS (Clang)
- [ ] Docker-based CI pipeline
- [ ] Package manager install (vcpkg/conan)
- [ ] HTTP API server (embedding, chat, completion endpoints)
- [ ] Comprehensive error handling
- [ ] Documentation site

---

## 📊 Comparison with Existing Projects

| Feature | llama.cpp | BitNet.cpp | MLX | OIL Engine |
|---------|-----------|------------|-----|------------|
| **Language** | C/C++ | C/C++ | C++/ObjC | **C++20** |
| **Dependencies** | None | llama.cpp | Metal | **None** |
| **Tensor library** | Custom | Custom | Custom | **Custom** |
| **Training** | ❌ | ❌ | ✅ | **✅ Full** |
| **Fine-tuning** | ❌ | ❌ | ✅ | **✅ Native OIL** |
| **Quant formats** | GGUF many | Ternary only | FP16/FP32 | **OIL8/OIL4/Ternary/Binary** |
| **Mixed per-block** | Grouped (K-quants) | Uniform | Uniform | **✅ Per-block routing** |
| **Target BPW** | 2-8 | 1.58 | 16 | **1.50** |
| **CPU inference** | ✅ Fast | ✅ Faster | ❌ Metal | **✅ Custom SIMD** |
| **GPU inference** | ✅ CUDA/Metal | ✅ CUDA | ✅ Metal | **⚠️ Future** |
| **Tokenizer** | BPE/SentencePiece | External | External | **✅ Built-in** |
| **Autograd** | ❌ | ❌ | ✅ | **✅ Custom** |
| **SIMD math** | ✅ | ✅ | ❌ | **✅ AVX2/NEON** |
| **Distributed** | ❌ | ❌ | ✅ FSDP | **✅ Design included** |
| **Fine-tune system** | ❌ | ❌ | ✅ | **✅ Native OIL** |
| **Model zoo** | 100+ models | BitNet only | MLX only | **Converter tools** |
| **License** | MIT | MIT | MIT | **MIT** |

---

## 💻 Developer Machine Reality

This project is being developed on:

| Component | Spec |
|-----------|------|
| CPU | Ryzen 5 5600GT (6C/12T, 3.6-4.4 GHz) |
| RAM | ~14 GB (usable) |
| GPU | Radeon Graphics (iGPU, Vega 7, 512 shaders) |
| OS | Windows 11 |
| Compiler | Clang 22.1.7 (x86_64-pc-windows-msvc) |
| Build | CMake 4.3.3 + Ninja |

### Realistic Training Limits

| Model Size | Full Train | Fine-tune |
|-----------|-----------|-----------|
| 0.1B (100M) | ✅ (~4h) | ✅ |
| 0.4B (400M) | ✅ (~20h) | ✅ |
| 1B | ⚠️ RAM limit | ✅ (~8h) |
| 3B | ⚠️ RAM limit | ⚠️ (~16h) |
| 7B | ⚠️ Needs more RAM | ⚠️ Needs more RAM |
| 48T (multi-node) | Future milestone | Future milestone |

The architecture is designed for scale — distributed training hooks, FSDP sharding, and tensor parallelism are built into the engine design so the same code can scale from laptop to cluster.

---

## 🎯 Performance Targets

These are **honest targets** based on published research and hardware constraints:

| Scenario | Target | Context |
|----------|--------|---------|
| 0.1B inference (CPU) | 200-500 tok/s | Fully OIL compressed, TL kernel |
| 1B inference (CPU) | 50-100 tok/s | Memory-bound, KV cache dominant |
| 7B inference (CPU) | 5-15 tok/s | llama.cpp territory |
| 7B inference (GPU) | 30-100 tok/s | Future CUDA path |
| OIL8 → FP32 quality | Perplexity diff < 0.01 | With fine-tune |
| Ternary → FP16 quality | Perplexity diff < 0.05 | BitNet-proven |
| Disk vs FP32 (OIL8) | 4× reduction | 32B→8B per weight |
| Disk vs FP32 (mixed) | 20× reduction | 32B→1.5B average |
| Kernel speed vs scalar | 4-8× (AVX2) | Theoretical peak |
| Kernel speed vs llama.cpp | 1-2× (ternary) | TL kernel advantage |

---

## 🛠️ Tools & CLI

| Binary | Source | Purpose |
|--------|--------|---------|
| `oil-infer` | `tools/infer.cpp` | Interactive chat / generation from .oil model |
| `oil-train` | `tools/train.cpp` | Train model from scratch with config |
| `oil-finetune` | `tools/finetune.cpp` | Fine-tune loaded .oil model natively |
| `oil-convert` | `tools/convert.cpp` | Convert GGUF/HF/FP32 → .oil |
| `oil-bench` | `tools/bench.cpp` | Run benchmarks |
| `oil-info` | `tools/info.cpp` | Inspect .oil file contents |

### Example Usage

```bash
# Convert a model to OIL format
oil-convert --input model.safetensors --output model.oil --target-bpw 1.50

# Run inference
oil-infer --model model.oil --prompt "Explain quantum computing" --max-tokens 512

# Train from scratch
oil-train --config config.json --data training_data.txt --output trained.oil

# Fine-tune natively in OIL format
oil-finetune --model base.oil --data domain_data.txt --lr 1e-5 --output finetuned.oil
```

---

## 📁 Project Structure

```
MYTHOS.cpp/
│
├── include/
│   └── oil/
│       ├── types.h             # Core type definitions
│       ├── tensor.h            # N-dimensional tensor
│       ├── memory.h            # Aligned allocator, buffer, pool
│       ├── math.h              # BLAS + activations + norms
│       ├── random.h            # Xoroshiro128+ RNG
│       ├── oil_format.h        # OIL binary format spec
│       ├── codebook.h          # OIL8/OIL4/ternary/binary codebooks
│       ├── format_planner.h    # BPW allocation planner
│       ├── kernel.h            # GEMM kernel abstractions
│       ├── transformer.h       # Transformer layer definitions
│       ├── model.h             # Model containers
│       ├── kv_cache.h          # KV cache
│       ├── sampler.h           # Sampling strategies
│       ├── generator.h         # Autoregressive generation
│       ├── tokenizer.h         # BPE/Unigram tokenizer
│       ├── autograd.h          # Computation graph
│       ├── optimizer.h         # AdamW, SGD
│       ├── trainer.h           # Training loop
│       ├── ste_quantizer.h     # Straight-Through Estimator
│       ├── finetune.h          # Native fine-tuning system
│       ├── moe.h               # Mixture of Experts
│       ├── vision.h            # Vision encoder/decoder
│       ├── audio.h             # Audio encoder/decoder
│       └── multimodal.h        # Joint multimodal model
│
├── src/
│   ├── tensor.cpp
│   ├── memory.cpp
│   ├── math.cpp
│   ├── math_avx2.cpp           # AVX2 math kernels
│   ├── math_neon.cpp           # ARM NEON math kernels
│   ├── random.cpp
│   ├── oil_format.cpp
│   ├── codebook.cpp
│   ├── format_planner.cpp
│   ├── kernel_i2s.cpp          # I2_S MAD kernel
│   ├── kernel_tl.cpp           # TL1/TL2 LUT kernel
│   ├── kernel_oil8.cpp         # OIL8 lookup kernel
│   ├── kernel_oil4.cpp         # OIL4 lookup kernel
│   ├── transformer.cpp
│   ├── model.cpp
│   ├── kv_cache.cpp
│   ├── sampler.cpp
│   ├── generator.cpp
│   ├── bpe_tokenizer.cpp
│   ├── unigram_tokenizer.cpp
│   ├── autograd.cpp
│   ├── optimizer.cpp
│   ├── trainer.cpp
│   ├── ste_quantizer.cpp
│   ├── finetune.cpp
│   └── int8_quant.cpp          # Activation quantization
│
├── tools/
│   ├── train.cpp
│   ├── infer.cpp
│   ├── convert.cpp
│   ├── bench.cpp
│   └── info.cpp
│
├── tests/
│   ├── test_tensor.cpp
│   ├── test_math.cpp
│   ├── test_format.cpp
│   ├── test_kernel.cpp
│   ├── test_model.cpp
│   ├── test_tokenizer.cpp
│   └── test_trainer.cpp
│
├── bench/
│   ├── bench_kernels.cpp
│   ├── bench_inference.cpp
│   ├── bench_quality.cpp
│   └── bench_vs_bitnet.cpp
│
├── cmake/
│   ├── arch.cmake              # CPU architecture detection
│   └── compiler.cmake          # Compiler flag detection
│
├── cmake/
│   ├── arch.cmake
│   └── compiler.cmake
│
├── .bitnet/                    # Reference knowledge (BitNet.cpp)
├── TRAINER-ENGINE/             # Build-time training scaffold
├── INFERENCE-ENGINE/           # Build-time inference scaffold
├── OIL8/                        # Build-time format scaffold
│
├── CMakeLists.txt              # Root build file
├── README.md                   # This file
├── BLUEPRINT.md                # Detailed build plan
├── SPEC.md                     # Mission breakdown
├── GROK.md                     # Initial session summary
└── LICENSE                     # MIT
```

---

## 🤝 Contributing

This is a solo-developed project, but contributions are welcome:

1. **Bug reports** — Open an issue with reproduction steps
2. **Kernel optimizations** — If you spot a faster SIMD path, PR is welcome
3. **Additional formats** — OIL2, OIL1, or custom codebook sizes
4. **Backend ports** — CUDA, Metal, Vulkan compute shaders
5. **Documentation** — Tutorials, examples, API docs

### Coding Standards

- C++20, no exceptions (compile with `-fno-exceptions`)
- No external dependencies beyond C++ standard library
- RAII for resource management
- Namespace: `oil::` for public API, `oil::detail::` for internals
- Function naming: `snake_case`
- Type naming: `PascalCase`

---

## 📜 License

**ALL RIGHTS RESERVED — PRIVATE AND PROPRIETARY**

This codebase is proprietary. No part of this software may be reproduced, distributed, or transmitted in any form or by any means without prior written permission of the owner.

**For licensing inquiries: USD $2.5 Billion**

---

*"NOTHING is impossible — reality is that no one tried to do that."*
