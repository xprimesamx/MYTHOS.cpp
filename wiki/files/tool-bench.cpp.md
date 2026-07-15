# `oil-bench` — Benchmarking CLI

**Path:** `tools/bench.cpp`

Performance benchmarking tool for kernels, inference, and model quality.

## Usage

```
oil-bench [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--kernels` | Benchmark compute kernels |
| `--inference` | Benchmark inference speed |
| `--quality` | Evaluate model quality (perplexity) |
| `--all` | Run all benchmarks |
| `--model` | Model path for quality benchmarks |
| `--dataset` | Dataset path for quality eval |

### Benchmark Types

| Benchmark | Measures | Output |
|-----------|----------|--------|
| Kernel | GFLOPS, bandwidth | MB/s, TFLOPS |
| Inference | tokens/second | t/s, latency/token |
| Quality | Perplexity | PPL on test set |

### Example

```bash
# Run all kernel benchmarks
oil-bench --kernels

# Inference benchmark
oil-bench --inference --model model.oil

# Quality evaluation
oil-bench --quality --model model.oil --dataset test.txt
```
