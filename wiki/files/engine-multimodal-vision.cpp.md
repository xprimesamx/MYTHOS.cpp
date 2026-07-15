# `engines/trainer/moe/vision/vision.cpp` — Vision Module

**Path:** `engines/trainer/moe/vision/vision.cpp`

Vision processing MoE expert module for image understanding.

## VisionExpert

```cpp
class VisionExpert {
    std::unique_ptr<ImageEncoder> encoder;
    
    Tensor process(const Tensor& image_input);
    Tensor encode_image(const Tensor& pixels);
    Tensor extract_features(const Tensor& encoded);
    std::vector<Tensor> get_patches(const Tensor& image);
};
```

### Features

- Vision Transformer (ViT) patch processing
- Image-to-feature encoding
- Object detection and classification
- Cross-modal image-text matching

### Input Format

- RGB image, normalized to [0, 1]
- Resized to model input size (e.g., 224×224)
- Optional center crop or resize

### MoE Integration

Vision expert is one of the specialized MoE experts, activated when the router detects image-related inputs.
