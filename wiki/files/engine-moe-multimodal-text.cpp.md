# `engines/trainer/moe/text/multimodal_text.cpp` — Multimodal Text Expert

**Path:** `engines/trainer/moe/text/multimodal_text.cpp`

Multimodal text processing MoE expert for cross-modal text understanding.

## MultimodalTextExpert

```cpp
class MultimodalTextExpert : public Expert {
    std::unique_ptr<TextEncoder> encoder;
    
    Tensor process_text(const Tensor& text);
    Tensor fuse_with_image(const Tensor& text, const Tensor& image_features);
    Tensor fuse_with_audio(const Tensor& text, const Tensor& audio_features);
};
```

## Operations

| Operation | Description |
|-----------|-------------|
| `process_text()` | Standard text encoding |
| `fuse_with_image()` | Text + image cross-attention |
| `fuse_with_audio()` | Text + audio cross-attention |

## Cross-Modal Fusion

Combines text features with other modality features using cross-attention mechanisms before passing to downstream tasks.
