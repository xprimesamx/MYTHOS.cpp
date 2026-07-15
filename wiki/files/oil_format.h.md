# `oil_format.h` — OIL Format I/O

**Path:** `include/oil/oil_format.h`

Reader and writer for the OIL binary model format — the single source of truth for model storage.

## OILReader

```cpp
class OILReader {
    OILReader(const std::string& path);
    
    OILHeader read_header();
    TransformerConfig read_config();
    Tensor read_tensor(const std::string& name);
    std::vector<std::string> list_tensors();
    bool has_tensor(const std::string& name) const;
    void close();
};
```

## OILWriter

```cpp
class OILWriter {
    OILWriter(const std::string& path, const TransformerConfig& config);
    
    void write_tensor(const std::string& name, const Tensor& tensor);
    void write_config(const TransformerConfig& config);
    void write_tokenizer(const std::vector<std::string>& vocab);
    void finalize();
    void close();
};
```

## OILHeader

```cpp
struct OILHeader {
    char magic[4];       // "OIL\0"
    uint32_t version;
    uint64_t num_tensors;
    uint64_t metadata_offset;
    uint64_t tensor_offset;
    uint64_t tokenizer_offset;
};
```

### File Structure

```
┌─────────────────┐
│ OILHeader        │ ← Magic, version, tensor count, offsets
├─────────────────┤
│ Model Config     │ ← Transformer architecture config
├─────────────────┤
│ Tensor Metadata  │ ← Names, shapes, dtypes, offsets
├─────────────────┤
│ Weight Data      │ ← Quantized weight tensors
├─────────────────┤
│ Tokenizer Data   │ ← BPE vocab & merges (optional)
└─────────────────┘
```

### Supported Operations

- Read/write model weights in all formats (FP32, FP16, OIL4, OIL8, Binary, Ternary)
- Forward-compatible versioning
- Memory-mapped reading for fast loading
- Partial tensor loading (lazy)
