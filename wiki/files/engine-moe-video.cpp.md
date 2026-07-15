# `engines/trainer/moe/video/video.cpp` — Video Expert

**Path:** `engines/trainer/moe/video/video.cpp`

Video processing MoE expert for video understanding and temporal modeling.

## VideoExpert

```cpp
class VideoExpert : public Expert {
    std::unique_ptr<VideoEncoder> encoder;
    std::unique_ptr<TemporalModel> temporal;
    
    Tensor process_video(const Tensor& frames);
    Tensor extract_temporal_features(const Tensor& frame_sequence);
    Tensor classify_video(const Tensor& video_tensor);
};
```

## Features

- Frame-by-frame feature extraction
- Temporal modeling (3D conv / attention over frames)
- Video classification and understanding
- Frame sampling for efficient processing (e.g., 1fps sampling)

## Input Format

- Video as sequence of frames [T, H, W, C]
- T frames sampled from video
- Each frame resized to model input size
