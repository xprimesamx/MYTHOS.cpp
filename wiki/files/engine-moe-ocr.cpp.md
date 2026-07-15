# `engines/trainer/moe/ocr/ocr.cpp` — OCR Expert

**Path:** `engines/trainer/moe/ocr/ocr.cpp`

Optical Character Recognition MoE expert for text extraction from images.

## OCRExpert

```cpp
class OCRExpert : public Expert {
    std::unique_ptr<VisionEncoder> vision_encoder;
    std::unique_ptr<TextDecoder> text_decoder;
    
    Tensor detect_text(const Tensor& image);
    Tensor recognize_text(const Tensor& text_region);
    std::vector<TextRegion> ocr_pipeline(const Tensor& document_image);
};
```

## Pipeline

```
Image → Text Detection → Region Crops → Recognition → Text Output

1. Detect text regions in image (bounding boxes)
2. Extract each region
3. Recognize characters in each region
4. Output structured text with positions
```

## Features

- Scene text detection in natural images
- Document OCR for scanned documents
- Handwriting recognition (when trained)
- Bounding box output with confidence scores
