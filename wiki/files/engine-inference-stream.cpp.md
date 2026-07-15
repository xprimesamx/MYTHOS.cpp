# `engines/inference/stream.cpp` — Stream Processing

**Path:** `engines/inference/stream.cpp`

Token streaming and interactive mode for the inference engine.

## StreamProcessor

```cpp
class StreamProcessor {
    using TokenCallback = std::function<void(const std::string&)>;
    using CompleteCallback = std::function<void(const std::string&)>;
    
    void set_token_callback(TokenCallback cb);
    void set_complete_callback(CompleteCallback cb);
    void process_token(int64_t token_id, bool is_final = false);
    void start_stream();
    void end_stream();
};
```

### Streaming Features

| Feature | Description |
|---------|-------------|
| Real-time token output | Tokens sent as generated |
| Buffered decoding | Efficient batch decode |
| Interactive mode | User input → model response loop |
| Stop conditions | EOS, max tokens, user interrupt |

### Interactive Mode Flow

```
User Input → Tokenize → Generate → Decode → Print → [loop]
                     ↓
              StreamProcessor
                     ↓
            Callback → stdout
```
