# `tokenizer.h` — BPE Tokenizer

**Path:** `include/oil/tokenizer.h`

Byte-Pair Encoding (BPE) tokenizer for encoding text to token IDs and decoding back.

## Tokenizer Class

```cpp
class Tokenizer {
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::pair<std::string, int>> merges;
    
    Tokenizer(const std::string& vocab_path);
    Tokenizer(const std::vector<std::string>& vocab);
    
    std::vector<int64_t> encode(const std::string& text);
    std::string decode(const std::vector<int64_t>& ids);
    int vocab_size() const;
    int bos_token() const;
    int eos_token() const;
    int pad_token() const;
};
```

### Operations

| Method | Description |
|--------|-------------|
| `encode(text)` | Tokenize text → token IDs (with BPE merging) |
| `decode(ids)` | Detokenize IDs → string |
| `vocab_size()` | Vocabulary size |
| `bos/eos/pad_token()` | Special token IDs |

### BPE Algorithm

1. Pre-tokenize input into bytes
2. Lookup each byte in vocabulary
3. Apply learned merge rules (highest priority first)
4. Continue until no more merges apply
5. Return resulting token IDs

### Special Tokens

- `<s>` — Beginning of sequence
- `</s>` — End of sequence
- `<pad>` — Padding
- `<unk>` — Unknown token
