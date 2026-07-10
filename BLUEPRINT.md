# MYTHOS.cpp — Complete Build Blueprint

## Identity
**Zero external dependencies.** Pure C++ AI engine. Training + Inference + Fine-tuning in one binary format (`.oil`). Every math op, every kernel, every tokenizer from scratch. No Python, no PyTorch, no HuggingFace.

---

## 1. Build System & Config

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Root — subdirs: core, inference, trainer, convert, bench, tools |
| `cmake/arch.cmake` | CPU detection (AVX2/AVX512/NEON, x86/ARM) |
| `cmake/compiler.cmake` | Compiler flags (Clang-cl/GCC/MSVC) |
| `cmake/config.cmake.in` | `oil_config.h` template — platform, SIMD level, debug flags |
| `oil_config.h` | Generated: `OIL_AVX2`, `OIL_DEBUG`, `OIL_VERSION` etc. |

---

## 2. Core Library (`liboil-core`)

### 2.1 Types (`include/oil/types.h`)
```
oil::Format   enum: BINARY(1), TERNARY(1.58), OIL4(4), OIL8(8), FP16(16), FP32(32)
oil::Shape    n-dim shape {rank, dims[]}
oil::DType    data-type for raw storage: u8, u4-packed, i2-packed, f16, f32
oil::Status   result type (OK / error string)
oil::Config   global engine flags (num_threads, seed, pool_size)
```

### 2.2 Memory (`include/oil/memory.h`, `src/memory.cpp`)
```
oil::AlignedAllocator   64-byte aligned malloc/free (SIMD-safe)
oil::Buffer             ref-counted byte buffer + alignment
oil::MemoryPool         arena allocator for small/temp tensors
```

### 2.3 Tensor (`include/oil/tensor.h`, `src/tensor.cpp`)
```
oil::Tensor
  .shape() .dtype() .format() .buffer()
  .view() .slice() .reshape() .transpose() .permute()
  .copy_to() .clone() .fill()
  .requires_grad() .grad() .backward()
  serialise/deserialise  (↔ .oil bytes)

oil::TensorOps
  .from_vector() .from_scalar() .zeros() .ones() .randn()
  .cat() .stack() .split()
```

### 2.4 Math Primitives (`include/oil/math.h`, `src/math.cpp`)
```
── BLAS ──
gemv(A, x, y)        y = α·A·x + β·y
gemm(A, B, C)        C = α·A·B + β·C
dot(x, y)            sum(x[i]·y[i])
axpy(a, x, y)        y[i] += a·x[i]

── Pointwise ──
relu/silu/gelu/sigmoid/tanh
mul/add/sub/div
exp/log/pow/sqrt

── Reduce ──
sum/mean/max/min    (along axis or all)
softmax             (stable: subtract max)
layer_norm/rms_norm

── Ternary/Binary ──
ternary_gemm(W_ternary, x, scale)    pack→add, no multiply
binary_gemm(W_binary, x, scale)      xor→popcount
oil8_gemm(W_idx, codebook, x)        gather→accumulate
oil4_gemm(W_idx, codebook, x)        gather→f16cvt→accumulate

── SIMD Flavours ──
  _avx2()    _avx512()    _neon()    _scalar()
  Selected at compile time via OIL_SIMD_LEVEL
```

### 2.5 Random (`include/oil/random.h`, `src/random.cpp`)
```
oil::RNG          Xoroshiro128+ (fast, deterministic)
  .uniform()      [0,1) f32
  .normal()       Box-Muller
  .uniform_int()  [lo, hi)
  .seed()         set/reset
```

---

## 3. OIL Format System (`liboil-format`)

### 3.1 Codebook (`include/oil/format/codebook.h`)
```
oil::CodebookU8    256 × f32 centroids    ─── OIL8
oil::CodebookU4    16  × f16 centroids    ─── OIL4
oil::CodebookT3    {neg, zero, pos} scale ─── Ternary
oil::CodebookB1    {neg, pos} scale       ─── Binary

Methods:
  .train(data)      k-means / EMA on weight block
  .quantize(w) → idx   nearest-centroid lookup
  .dequantize(idx) → f32
  .serialise() / .deserialise()
```

### 3.2 `.oil` Binary Layout
```
┌─ FileHeader (64 B) ──────────────────────┐
│ magic="OIL1"  version  flags  model_meta  │
├─ FormatTable ─────────────────────────────┤
│ per-block: {block_id, Format, codebook_sz}│
├─ Block Data ──────────────────────────────┤
│ block_0: codebook | packed_indices        │
│ block_1: codebook | packed_indices        │
│ ...                                       │
├─ Tensor Names ────────────────────────────┤
│ name_0 → block_0:block_2                  │
│ name_1 → block_3                          │
└───────────────────────────────────────────┘
```

### 3.3 Serialiser/Deserialiser
```
oil::OILWriter(path)     create/append .oil
oil::OILReader(path)     read .oil, iterate blocks/tensors
oil::OILValidator(path)  checksum + format validity
```

### 3.4 Format Planner
```
oil::FormatPlanner
  .score_importance(model, calibration_data)
  .allocate(target_bpw=1.50)
    1. Find 1% most salient weights → assign OIL8 (8b)
    2. Next 4% important → OIL4 (4b)
    3. Bulk → ternary (1.58b) or binary (1.0b)
    4. Compute average BPW
    5. If >1.50, shift boundary: more → binary
  .export_plan() → FormatTable
```

---

## 4. Model Architecture (`liboil-model`)

### 4.1 Config
```
oil::ModelConfig
  {arch, vocab, hidden, layers, heads, head_dim,
   ffn_hidden, activation, norm_eps, rope_theta,
   max_seq_len, moe_num_experts, moe_top_k}
```

### 4.2 Layers
```
oil::Linear         W(format_matrix) + bias
oil::Embedding      token → f32 lookup
oil::RMSNorm        x * rsqrt(mean(x²) + ε)
oil::LayerNorm      (x - μ) / σ * γ + β
oil::RotaryEmbedding    cos/sin per head
oil::Attention      QKV → score → softmax → output
oil::FFN            up/gate/down (SwiGLU)
oil::MoERouter      top-k routing + load-balancing loss
oil::MoEFFN         N experts, each = FFN
oil::TransformerBlock   Attn + FFN + norms + residual
```

### 4.3 Models
```
oil::DenseModel       { embeddings + N×transformer_block + lm_head }
oil::MoEModel         { embeddings + N×(attn + moe_ffn) + lm_head }
oil::MultimodalModel  { text_encoder, vision_encoder, cross_attn, ... }
```

All models implement:
```
.load(oil_file)        load from .oil
.save(oil_file)        save to .oil (OIL format weights)
.forward(input_ids)    logits output
.generate(config)      auto-regressive
```

---

## 5. INFERENCE-ENGINE (`liboil-inference`)

### 5.1 Context
```
oil::InferenceConfig     temperature, top_k, top_p, rep_penalty, max_tokens
oil::InferenceState      KV cache buffer, current seq position
```

### 5.2 KV Cache
```
oil::KVCache
  .append(k, v)
  .get(pos) → {k, v}
  .clear()
  Supports OIL4 compressed KV (BitNet style)
```

### 5.3 Sampling
```
oil::Sampler
  .greedy(logits) → token_id
  .top_k(logits, k) → token_id
  .top_p(logits, p) → token_id
  .beam_search(model, prefix, beams, len) → sequences
```

### 5.4 Decoding Loop
```
oil::Generator
  .generate(prompt_ids, config) → output_ids
  .stream(prompt_ids, config, on_token_callback)
```

---

## 6. TRAINER-ENGINE (`liboil-trainer`)

### 6.1 Autograd
```
oil::autograd::Graph        DAG of computation nodes
oil::autograd::Node         tensor → op → tensors
oil::autograd::Function     base: forward() + backward()

Each op registered:
  MatMulGrad
  AddGrad / SubGrad / MulGrad
  ReLUGrad / SiLUGrad / GELUGrad
  SoftmaxGrad
  LayerNormGrad / RMSNormGrad
  CrossEntropyGrad
```

### 6.2 Optimisers
```
oil::SGD(lr, momentum, weight_decay)
oil::AdamW(lr, betas, eps, weight_decay)
oil::Adam
  .step()          apply gradients → update params
  .zero_grad()     reset gradients
  .lr_scheduler    cosine / linear / warmup
  .clip_grad_norm(max_norm)
```

### 6.3 OIL-Native Training
```
oil::STEQuantizer
  Forward:  quantise weights (ternary/binary/OIL4/OIL8)
  Backward: straight-through (gradients pass through unchanged)

oil::CodebookUpdater
  After each step, update codebook centroids via EMA (moving average)

oil::QuantAwareTrainer
  Wraps any model with STE + codebook update
  Training loop: forward(quant) → loss → backward → optim(FP32) → codebook_update
```

### 6.4 LoRA
```
oil::lora::Config        rank, alpha, target_modules, dropout
oil::lora::Linear        wraps Linear: output = W·x + α/r · A·B·x
oil::lora::Optimiser     only LoRA params have grad, base frozen
oil::lora::Merge         fuse adapters back into base weights
```

### 6.5 Training Loop
```
oil::Trainer
  .compile(model, loss_fn, optimiser)
  .fit(dataloader, epochs, callbacks)
  .save_checkpoint(path)    model + optimiser state → .oil
  .load_checkpoint(path)    resume training

oil::DataLoader
  .from_text(file)          tokenize on the fly
  .batch(batch_size, seq_len) → {input_ids, labels}
  .shuffle() .repeat()

oil::Evaluator
  .perplexity(model, dataset)
  .accuracy(model, dataset)
```

### 6.6 Distributed (Scale Design)
```
oil::dist::Config     world_size, rank, backend
oil::dist::AllReduce  gradient sync across ranks
oil::dist::FSDP       shard model params + gather on forward
oil::dist::TP         tensor parallelism for huge layers
```

---

## 7. Tokenizer (`liboil-tokenizer`)

```
oil::BPETokenizer
  .train(files, vocab_size)      learn merges
  .encode(text) → ids
  .decode(ids) → text
  .save(path) / .load(path)      .oil tokenizer files

oil::UnigramTokenizer
  .train(files, vocab_size)      EM training
  .encode() .decode()

oil::TokenizerConfig
  {type, vocab_size, bos_id, eos_id, pad_id, unk_id}
```

---

## 8. Converters (`liboil-convert`)

```
oil::convert::from_gguf(gguf_path, oil_path, plan)
    Load GGUF → read weights → apply FormatPlanner → write .oil

oil::convert::from_safetensors(hf_dir, oil_path, config, plan)
    Read model.safetensors + config.json → plan → write .oil

oil::convert::from_fp32(raw_path, oil_path, plan)
    Raw f32 weights → plan → .oil

oil::convert::to_fp32(oil_path, output_dir)
    Decompress .oil back to f32 for verification
```

---

## 9. Command-Line Tools (`tools/`)

| Binary | Source | Purpose |
|--------|--------|---------|
| `oil-train` | `tools/train.cpp` | Train model from scratch or resume |
| `oil-infer` | `tools/infer.cpp` | Interactive chat / generation |
| `oil-convert` | `tools/convert.cpp` | Convert models → .oil |
| `oil-bench` | `tools/bench.cpp` | Benchmark kernels, throughput, quality |
| `oil-info` | `tools/info.cpp` | Inspect .oil file contents |
| `oil-finetune` | `tools/finetune.cpp` | LoRA fine-tune a .oil model |

---

## 10. Benchmarks (`bench/`)

```
bench_kernels.cpp      matmul, gemm, norm throughput (vs scalar baseline)
bench_inference.cpp    tok/s, memory usage, KV cache perf
bench_quality.cpp      perplexity comparison (FP32 vs OIL8 vs OIL4 vs ternary)
bench_vs_bitnet.cpp    ternary speed matching BitNet's 2-6x claims
bench_vs_ggml.cpp      inference speed comparison
```

---

## 11. Tests (`tests/`)

```
test_tensor.cpp        shape, view, slice, reshape, serialise round-trip
test_math.cpp          gemm correctness, gradient check
test_format.cpp        encode→decode→equality for each format
test_plan.cpp          FormatPlanner targeting 1.50 BPW
test_model.cpp         tiny model forward/backward, gradient numerical check
test_lora.cpp          LoRA forward/backward, merge, unmerge
test_kv_cache.cpp      append/get/clear correctness
test_tokenizer.cpp     encode→decode identity, BPE merge correctness
test_inference.cpp     generate output matches expected
test_converter.cpp     FP32 → OIL8 → FP32, check max error
```

---

## 12. Build Order (Execution)

```
Phase 1 — Core Foundation (current)
  1.1  CMake project + platform detection
  1.2  types.h + oil_config.h
  1.3  memory.h → AlignedAllocator + Buffer
  1.4  tensor.h / tensor.cpp  (full Tensor class)
  1.5  math.h / math.cpp  (scalar + AVX2 paths)
  1.6  random.h / random.cpp
  1.7  format/codebook.h (OIL8 + OIL4 + Ternary + Binary)
  1.8  format/serialiser.h (OILWriter + OILReader)
  1.9  format/planner.h (BPW allocator)
  1.10 test: tensor round-trip, math correctness, format encode→decode

Phase 2 — Inference
  2.1  model config + layer classes (Linear, RMSNorm, RoPE, Attn, FFN)
  2.2  model container (DenseModel)
  2.3  OIL8/OIL4 gemm kernels (AVX2 + scalar)
  2.4  ternary gemm kernel (from .bitnet knowledge)
  2.5  KV cache
  2.6  sampler + generator loop
  2.7  tokenizer (BPE)
  2.8  converter (FP32 → .oil conversion)
  2.9  tools/infer.cpp CLI
  2.10 test: load small model, generate tokens

Phase 3 — Training
  3.1  autograd graph + Function base
  3.2  matmul + norm + softmax + activations gradients
  3.3  cross-entropy loss gradient
  3.4  AdamW optimiser
  3.5  Trainer loop + DataLoader
  3.6  Checkpoint save/load
  3.7  STE quantiser + codebook update
  3.8  LoRA adapter system
  3.9  tools/train.cpp + tools/finetune.cpp
  3.10 test: train tiny model, verify loss decreases

Phase 4 — Scale & Multimodal
  4.1  MoE layers (router, experts, load balancing)
  4.2  Distributed primitives (AllReduce, FSDP design)
  4.3  Vision encoder/decoder
  4.4  Audio encoder/decoder
  4.5  Video encoder/decoder
  4.6  OCR module
  4.7  MultimodalModel
  4.8  Full benchmark suite
```

---

## Totals (Estimated Lines of Code)

| Module | Files | Est. LOC |
|--------|-------|----------|
| Core (types, memory, tensor) | 6 | 3,000 |
| Math (BLAS, pointwise, kernels) | 8 | 5,000 |
| OIL Format (codebook, serial, planner) | 6 | 3,500 |
| Model Architecture (layers, models) | 10 | 6,000 |
| Inference (KV cache, sampler, generator) | 5 | 2,500 |
| Tokenizer | 3 | 2,500 |
| Autograd + Ops | 8 | 4,000 |
| Optimisers + Trainer | 6 | 3,000 |
| STE + LoRA + Quant-aware | 4 | 2,000 |
| Distributed | 3 | 1,500 |
| Converters | 4 | 2,000 |
| Tools (CLI) | 5 | 2,000 |
| Tests | 15 | 5,000 |
| Benchmarks | 5 | 1,500 |
| Build system | 5 | 500 |
| **Total** | **93** | **~44,000** |
