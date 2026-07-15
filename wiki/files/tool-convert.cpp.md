# `oil-convert` — Model Conversion CLI

**Path:** `tools/convert.cpp`

Command-line tool for converting HuggingFace/SAFETensors models to OIL format.

## Usage

```
oil-convert --input <model.safetensors> [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | Input model path (safetensors) |
| `--output` | required | Output OIL file path |
| `--config` | - | Model config JSON |
| `--target-bpw` | 16.0 | Target bits-per-weight |
| `--format` | auto | Force format (fp16/oil4/oil8) |
| `--tokenizer` | - | Tokenizer vocab file |
| `--verbose` | false | Verbose output |

### Conversion Process

1. Read input safetensors
2. Parse model configuration
3. Quantize weights to target format (if BPW < 16)
4. Write OIL format with metadata
5. (Optional) Embed tokenizer data

### Example

```bash
# Full precision conversion
oil-convert --input model.safetensors --output model.oil

# Quantized conversion
oil-convert --input model.safetensors --output model.oil --target-bpw 1.50
```
