# `log_writer.cpp` — Logger Implementation

**Path:** `src/log_writer.cpp`

Training metrics logger: scalars, text, and config persistence.

## Functions

| Function | Description |
|----------|-------------|
| `log_scalar(name, value, step)` | Append scalar to events log |
| `log_dict(dict, step)` | Log multiple metrics at once |
| `log_text(tag, text)` | Log sample outputs |
| `save_config(json)` | Save experiment config |
| `flush()` | Force write to disk |

## Log Format

JSON Lines (JSONL) format:
```json
{"type": "scalar", "name": "loss", "value": 2.34, "step": 100}
{"type": "scalar", "name": "lr", "value": 3e-4, "step": 100}
{"type": "text", "tag": "sample", "text": "Generated output", "step": 100}
```

## Directory Structure

```
logs/experiment_name/
├── config.json        # Experiment config
├── events.jsonl       # Time-series metrics
├── train.log          # Console output log
└── checkpoints/       # Model checkpoints
    ├── step_1000.oil
    ├── step_2000.oil
    └── best.oil
```
