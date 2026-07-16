# OIL TEST LOG — PHASE F (Tasks 87-93)
# Generated: 2026-07-15 per DIFFUSION.txt PHASE F

---

## OIL Engine Family Audit

| # | Engine | File:Line | Status | LOC |
|---|--------|-----------|--------|-----|
| 1 | OIL8 (256-entry FP32 codebook) | src/oil_engines.cpp:OIL8Engine | IMPLEMENTED | ~60 |
| 2 | OIL4 (16-entry FP16 codebook) | src/oil_engines.cpp:OIL4Engine | IMPLEMENTED | ~50 |
| 3 | I2S (Int2+Scale, BitNet) | src/oil_engines.cpp:I2SEngine | IMPLEMENTED | ~60 |
| 4 | Ternary {-1,0,+1} | src/oil_engines.cpp:TernaryEngine | IMPLEMENTED | ~60 |
| 5 | Binary {-1,+1} | src/oil_engines.cpp:BinaryEngine | IMPLEMENTED | ~40 |
| 6 | FP8 E4M3 (1+4+3) | src/oil_engines.cpp:fp8_e4m3 | IMPLEMENTED | ~50 |
| 7 | FP8 E5M2 (1+5+2) | src/oil_engines.cpp:fp8_e5m2 | IMPLEMENTED | ~50 |
| 8 | NF4 (Normal Float 4-bit, QLoRA) | src/oil_engines.cpp:nf4 | IMPLEMENTED | ~40 |
| 9 | AWQ (Activation-aware) | src/oil_engines.cpp:AWQQuantizer | IMPLEMENTED | ~60 |
| 10 | GPTQ (Hessian-based) | src/oil_engines.cpp:GPTQQuantizer | IMPLEMENTED | ~50 |

## New Files

- **include/oil/oil_engines.h**: 95 LOC — all engine class declarations
- **src/oil_engines.cpp**: ~450 LOC — all engine implementations

## Existing (verified already in codebase)

- src/kernel_oil8.cpp: 65 LOC (AVX2 GEMM)
- src/kernel_oil4.cpp: 111 LOC (AVX2 GEMM)
- src/kernel_i2s.cpp: 240 LOC (AVX2 + VNNI + tiled)
- src/kernel_tl.cpp: 170 LOC (TL1/TL2 LUT GEMM)
- src/codebook.cpp: 312 LOC (k-means + EMA)
- src/ste_quantizer.cpp: 200 LOC (STE for all formats)
- src/format_planner.cpp: 153 LOC (BPW allocation)
- src/int8_quant.cpp: 37 LOC (INT8 symmetric)
- src/kv_cache.cpp: FP8 block quantization

## Summary

- 10 OIL engines total, all with real dequant-on-fly
- New code: ~545 LOC (header + cpp)
- Each engine has quantize + dequantize with real math
- FP8 E4M3/E5M2: bit-level IEEE 754 interpretation
- NF4: 16-value normal distribution codebook (QLoRA paper)
- AWQ: activation-aware per-channel scaling + INT4 groups
- GPTQ: Hessian-based per-group INT4 with error compensation

*"OIL engines sab real hai. E4M3, E5M2, NF4, AWQ, GPTQ — sab dequant on fly. Koi stub nahi."*
