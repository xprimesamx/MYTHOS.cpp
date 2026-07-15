# `oil-train` — Training CLI

**Path:** `tools/train.cpp`

Command-line tool for training models from scratch.

## Usage

```
oil-train --config <config.json> --data <data.txt> --output <model.oil> [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | Model & training config JSON |
| `--data` | required | Training data file path |
| `--output` | required | Output OIL model path |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 8 | Batch size |
| `--epochs` | 10 | Training epochs |
| `--resume` | - | Resume from checkpoint |
| `--log-dir` | logs/ | Log directory |

### Example

```bash
oil-train --config config.json --data data.txt --output trained.oil
```
