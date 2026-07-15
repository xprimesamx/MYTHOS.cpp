# `multimodal.cpp` — Multi-Modal Implementation

**Path:** `src/multimodal.cpp`

Multi-modal model support: text, image, and audio processing.

## Functions

| Function | Description |
|----------|-------------|
| `encode_image(pixels)` | Run ViT encoder on image |
| `encode_audio(waveform)` | Run audio encoder |
| `fuse(features)` | Cross-modal feature fusion |

## Forward Pass (Multimodal)

```cpp
Tensor MultimodalModel::forward(input_ids, positions, cache) {
    // 1. Text embedding
    Tensor text_feat = tok_embeddings->forward(input_ids);
    // 2. If vision features present, fuse
    if (vision_features) {
        Tensor vision_feat = encode_image(*vision_input);
        text_feat = fuse(text_feat, vision_feat);
    }
    // 3. Standard transformer forward
    for (auto& layer : layers) {
        text_feat = layer->forward(text_feat, cache);
    }
    // 4. Final norm + lm_head
    return lm_head->forward(norm->forward(text_feat));
}
```

## Feature Fusion

Multi-modal fusion uses cross-attention:
- Vision features as keys/values
- Text features as queries
- Output: vision-conditioned text features
