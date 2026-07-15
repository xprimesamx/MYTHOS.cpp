# `generator.cpp` — Text Generation Pipeline

**Path:** `src/generator.cpp`

Implements text generation: tokenization → inference loop → sampling → detokenization.

## Key Functions

| Function | Description |
|----------|-------------|
| `Generator::generate()` | Full text generation from prompt |
| `Generator::stream_generate()` | Token-by-token streaming |
| `Generator::generate_ids()` | ID-based generation |

## Generation Loop

```
1. Tokenize prompt → token_ids
2. Encode position IDs
3. Initialize KV cache
4. For step = 0..max_tokens:
   a. Model forward → logits
   b. Sample from logits → next_token
   c. If next_token == EOS → stop
   d. Append next_token to sequence
   e. Stream callback (if streaming)
5. Decode token_ids → text
```

## Memory Management

- KV cache reallocates as sequence grows
- Intermediate tensors freed between steps
- Output string built incrementally
