# `bpe_tokenizer.cpp` — BPE Tokenizer Implementation

**Path:** `src/bpe_tokenizer.cpp`

Byte-Pair Encoding tokenizer implementation.

## Tokenizer Initialization

```cpp
Tokenizer::Tokenizer(const std::string& vocab_path) {
    // Load vocabulary: token → id mapping
    // Load merge rules: sorted by priority
    // Set special tokens: BOS, EOS, PAD, UNK
}
```

## Encoding Algorithm

```
1. Normalize input text
2. Convert to bytes
3. Lookup each byte in vocab → initial tokens
4. While merge_rule applies:
   a. Find highest-priority merge in current sequence
   b. Replace pair with merged token
5. Return token ID sequence
```

## Decoding

```
1. For each token ID:
2.   Lookup token string in vocab
3.   Concatenate (handling BPE artifacts)
4. Return decoded text
```

## Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<s>` | 1 | Beginning of sequence |
| `</s>` | 2 | End of sequence |
| `<unk>` | 3 | Unknown token |
| `<pad>` | 0 | Padding |
