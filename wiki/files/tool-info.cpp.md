# `oil-info` — Model Info CLI

**Path:** `tools/info.cpp`

Inspect OIL model file metadata and statistics.

## Usage

```
oil-info --model <model.oil> [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | OIL model path |
| `--verbose` | false | Detailed output |
| `--summary` | false | Summary statistics only |

### Output

| Field | Description |
|-------|-------------|
| Model name | Name from config |
| Format | Weight format (FP16/OIL4/OIL8/etc.) |
| BPW | Bits-per-weight |
| Parameters | Total parameter count |
| Layers | Number of transformer layers |
| Hidden size | Model dimension |
| Heads | Number of attention heads |
| Vocabulary | Vocabulary size |
| File size | OIL file size |
| Compression ratio | vs FP32 baseline |

### Example

```bash
oil-info --model model.oil
# Output:
# Model: my-model
# Format: OIL4 (1.50 BPW)
# Parameters: 7,000,000,000
# File size: 1.31 GB
# Compression ratio: 21.3x vs FP32
```
