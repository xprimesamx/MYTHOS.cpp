# `oil-infer` — Inference CLI

**Path:** `tools/infer.cpp`

Command-line tool for running inference with OIL models.

## Usage

```
oil-infer --model <model.oil> [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Path to OIL model file |
| `--prompt` | required | Input text prompt |
| `--max-tokens` | 256 | Maximum tokens to generate |
| `--temperature` | 1.0 | Sampling temperature |
| `--top-p` | 0.95 | Nucleus sampling |
| `--top-k` | 40 | Top-K sampling |
| `--seed` | -1 | Random seed |
| `--stream` | false | Stream output tokens |
| `--interactive` | false | Interactive chat mode |
| `--bench` | false | Benchmark only |

### Example

```bash
oil-infer --model model.oil --prompt "Hello" --max-tokens 256 --temperature 0.8
```
