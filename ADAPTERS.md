# ADAPTERS — External Format Isolation (PHASE J)

## Overview

MYTHOS.cpp core engine NEVER includes adapters. Adapters are one-way: they import external formats INTO `.oil` and export FROM `.oil` to external formats. Core `oil_format.cpp` does NOT depend on adapters.

## Architecture

```
src/adapters/
├── lora.cpp          — LoRAAdapter (low-rank adaptation)
├── qlora.cpp         — QLoRAAdapter (NF4 quantized base + LoRA)
├── dora.cpp          — DoRAAdapter (weight-decomposed LoRA)
├── gguf_import.cpp   — GGUFImporter (GGUF → .oil)
├── safetensors.cpp   — SafetensorsImporter (.safetensors → .oil)
└── include/oil/adapters.h — All adapter declarations
```

## Namespace

All adapters live in `mythos::adapters`. Core lives in `oil::`.

## Oil Guard

`mythos::adapters::oil_load(path)` checks if path ends with `.oil`. If it's `.gguf`, `.ggml`, or `.safetensors`, it throws an error directing the user to the appropriate importer. This prevents accidental loading of non-OIL formats.

## Adapters

### LoRAAdapter
- Low-rank decomposition: `output = W*x + (alpha/r) * B * A * x`
- `set_base(W)`: set frozen base weights
- `save/load`: serialize to .oil format
- File: src/adapters/lora.cpp

### QLoRAAdapter
- NF4-quantized base weights + LoRA adapters
- `quantize_base(W)`: quantize base to 4-bit codebook
- `dequantize_base()`: reconstruct FP32 approximation
- File: src/adapters/qlora.cpp

### DoRAAdapter
- Weight-decomposed low-rank adaptation
- Decomposes W into magnitude (per-row norm) + direction (normalized)
- `decompose(W)`: decompose base weights
- `forward(x)`: magnitude * direction * x + (alpha/r) * B * A * x
- File: src/adapters/dora.cpp

### GGUFImporter
- Reads GGUF v1-v3 binary format
- Parses metadata, tensor names, shapes, dtypes
- `import_gguf(in, out)`: converts to .oil, NEVER overwrites input
- File: src/adapters/gguf_import.cpp

### SafetensorsImporter
- Reads safetensors format (8-byte header size + JSON header + raw tensor data)
- Parses tensor names, shapes, data offsets
- `import_safetensors(in, out)`: converts to .oil
- File: src/adapters/safetensors.cpp

## .gitignore

Blocks: `*.bin *.gguf *.ggml *.safetensors *.lora *.pt *.ckpt`
Allows: `dist/` (binary release directory)

## Usage

```cpp
#include "oil/adapters.h"

// Import GGUF to OIL
mythos::adapters::GGUFImporter::import_gguf("model.gguf", "model.oil");

// Import safetensors to OIL
mythos::adapters::SafetensorsImporter::import_safetensors("model.safetensors", "model.oil");

// Load the .oil file (core, no adapters needed)
auto reader = mythos::adapters::oil_load("model.oil");

// LoRA fine-tune
mythos::adapters::LoRAAdapter lora(768, 768, 16, 32.0f);
lora.set_base(base_weights);
auto adapted = lora.forward(input);
```
