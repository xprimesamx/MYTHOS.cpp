# `engines/trainer/moe/audio/audio.cpp` — Audio Module

**Path:** `engines/trainer/moe/audio/audio.cpp`

Audio processing MoE expert module for speech and audio understanding.

## AudioExpert

```cpp
class AudioExpert : public Expert {
    std::unique_ptr<AudioEncoder> encoder;
    std::unique_ptr<AudioDecoder> decoder;
    
    Tensor process(const Tensor& audio_input);
    Tensor encode_audio(const Tensor& waveform);
    Tensor decode_audio(const Tensor& features);
    int sample_rate() const;
};
```

### Features

- Mel-spectrogram feature extraction
- Audio classification and understanding
- Optional speech-to-text pipeline
- Audio embedding generation

### Input Format

- Raw waveform (PCM 16-bit, 16kHz)
- Normalized to [-1, 1] range
- Variable length, padded to max duration
