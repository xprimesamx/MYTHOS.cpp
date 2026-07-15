# `multimodal.h` — Multi-Modal Support

**Path:** `include/oil/multimodal.h`

Multi-modal model support for processing text, images, and audio inputs.

## MultimodalModel Class

```cpp
class MultimodalModel : public Model {
    std::unique_ptr<Embedding> text_embed;
    std::unique_ptr<ImageEncoder> image_encoder;
    std::unique_ptr<AudioEncoder> audio_encoder;
    std::unique_ptr<Projection> proj;
    
    Tensor forward(const Tensor& input_ids, const Tensor& positions,
                   KVCache* cache = nullptr) override;
    
    Tensor encode_image(const Tensor& image);
    Tensor encode_audio(const Tensor& audio);
    Tensor fuse(const Tensor& text_features, const Tensor& vision_features);
};
```

### Modality Encoders

| Encoder | Input | Output |
|---------|-------|--------|
| Text | token IDs | text embeddings |
| Image | pixel data | vision features |
| Audio | waveform | audio features |

### Multi-Modal Fusion

The fusion mechanism:
1. Encode each modality separately
2. Project to shared embedding space
3. Fuse via cross-attention layers
4. Process through shared transformer backbone
