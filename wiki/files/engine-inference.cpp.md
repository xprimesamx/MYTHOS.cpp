# `engine/inference.cpp` — Inference Engine

**Path:** `engines/inference/inference.cpp`

The main inference engine that runs trained OIL models for text generation.

## InferenceEngine

| Feature | Description |
|---------|-------------|
| Model loading | Load `.oil` model files |
| Tokenization | BPE tokenization of input |
| Forward pass | Run transformer forward |
| Sampling | Temperature, top-p, top-k |
| KV caching | Cache K,V for efficiency |
| Streaming | Token-by-token output |

### Forward Pass

1. Tokenize input prompt
2. Build position tensor
3. Initialize/populate KV cache
4. For each generation step:
   - Run attention (with KV cache)
   - Run feed-forward
   - Sample next token
5. Decode output tokens to text
6. Return/stream result

### Engine Components

| Component | File | Role |
|-----------|------|------|
| `InferenceEngine` | `inference.h/cpp` | Main inference orchestrator |
| `StreamProcessor` | `stream.cpp` | Token streaming |
| `KVCacheManager` | Built-in | KV cache lifecycle |
