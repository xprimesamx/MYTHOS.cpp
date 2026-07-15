# Modules

## Core Libraries (37 targets)

### Foundation Layer
| Module | Files | Description |
|--------|-------|-------------|
| **oil_core** | `tensor`, `autograd`, `memory`, `types`, `random` | Tensor with autograd, memory pool |
| **oil_types** | `types.h` | Format/DType enums, Shape, TransformerConfig |
| **oil_memory** | `memory.h/cpp` | Buffer allocation, pooling, cross-device copy |
| **oil_random** | `random.h/cpp` | Random number generation |
| **oil_tensor** | `tensor.h/cpp` | N-dimensional array |
| **oil_autograd** | `autograd.h/cpp` | Automatic differentiation |

### Math & Compute Layer
| Module | Files | Description |
|--------|-------|-------------|
| **oil_math** | `math.h`, `math.cpp`, `math_avx2.cpp` | Math ops with AVX2 |
| **oil_kernel** | `kernel.h`, `kernel_oil4.cpp`, `kernel_oil8.cpp`, `kernel_i2s.cpp`, `kernel_tl.cpp` | Quantization kernels |

### Model Layer
| Module | Files | Description |
|--------|-------|-------------|
| **oil_model** | `model.h/cpp` | Model interface |
| **oil_transformer** | `transformer.h/cpp` | Transformer architecture |
| **oil_backend** | `backend.h/cpp` | Device backend |
| **oil_tokenizer** | `tokenizer.h`, `bpe_tokenizer.cpp` | BPE tokenizer |
| **oil_format** | `oil_format.h/cpp` | OIL format I/O |

### Engine Layer
| Module | Files | Description |
|--------|-------|-------------|
| **oil_engine** | `engines/inference/` | Inference engine |
| **oil_trainer** | `src/trainer.cpp`, `engines/trainer/` | Training engine |
| **oil_oil8** | `engines/OIL8/` | OIL8 codec |
| **oil_dense** | `engines/trainer/dense/` | Dense training |
| **oil_moe** | `moe.h/cpp`, `engines/trainer/moe/` | MoE training |
| **oil_gpu** | `gpu_compute.h/cpp`, `gpu_extras.h/cpp` | GPU acceleration |

### Production Layer
| Module | Files | Description |
|--------|-------|-------------|
| **oil_inference_opt** | `inference_opt.h/cpp` | Optimized inference |
| **oil_generator** | `generator.h/cpp` | Text generation |
| **oil_sampler** | `sampler.h/cpp` | Token sampling |
| **oil_kv_cache** | `kv_cache.h/cpp` | KV cache |
| **oil_optimizer** | `optimizer.h/cpp` | AdamW/SGD |
| **oil_log_writer** | `log_writer.h/cpp` | Logging |

### Advanced Features
| Module | Files | Description |
|--------|-------|-------------|
| **oil_asi** | `asi.h/cpp`, `asi_pipeline.h/cpp` | ASI cognitive arch |
| **oil_flash_attention** | `flash_attention.h/cpp` | Flash attention |
| **oil_multimodal** | `multimodal.h/cpp` | Multi-modal support |
| **oil_ste_quantizer** | `ste_quantizer.h/cpp` | STE quantizer |
| **oil_codebook** | `codebook.h/cpp` | VQ codebooks |
| **oil_int8_quant** | `int8_quant.h/cpp` | INT8 quantization |
| **oil_format_planner** | `format_planner.h/cpp` | Format planning |
| **oil_production** | `production.h/cpp` | Production deployment |
| **oil_distributed** | `distributed.h/cpp` | Distributed training |
| **oil_finetune** | `finetune.h/cpp` | Fine-tuning |

### MoE Submodules
| Module | Files | Description |
|--------|-------|-------------|
| **oil_moe_core** | `moe.h`, `engines/trainer/moe/moe.cpp` | Core MoE |
| **oil_moe_enhance** | `moe_enhance.h/cpp` | Enhanced MoE |
| **oil_moe_variants** | `moe_variants.h/cpp` | MoE variants |
| **oil_moe_audio** | `engines/trainer/moe/audio/` | Audio expert |
| **oil_moe_vision** | `engines/trainer/moe/vision/` | Vision expert |
| **oil_moe_video** | `engines/trainer/moe/video/` | Video expert |
| **oil_moe_ocr** | `engines/trainer/moe/ocr/` | OCR expert |
| **oil_moe_embeddings** | `engines/trainer/moe/embeddings/` | Embedding expert |
| **oil_moe_text** | `engines/trainer/moe/text/` | Multimodal text |

## Complete File Index

See [File Documentation Index](files/_index) for per-file documentation of all modules.
