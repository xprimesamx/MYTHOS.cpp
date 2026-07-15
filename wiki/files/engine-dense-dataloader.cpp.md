# `engines/trainer/dense/dataloader.cpp` — Dense Dataloader

**Path:** `engines/trainer/dense/dataloader.cpp`

Data loading for training: file reading, tokenization, batch creation.

## Implementation

```cpp
DataLoader::DataLoader(path, batch_size, seq_len) {
    // 1. Read entire file into memory
    // 2. Tokenize all text
    // 3. Split into sequences of seq_len
    // 4. Group into batches
}

bool DataLoader::next(input, target) {
    // input  = tokens[batch_pos:batch_pos + seq_len]
    // target = tokens[batch_pos + 1:batch_pos + seq_len + 1]
    // (shifted by 1 for next-token prediction)
}
```

## Data Pipeline

```
Raw Text → Tokenize → [token_id, ...] → Sequences → Batches
                                                         ↓
                                              [B, seq_len] input
                                              [B, seq_len] target
```

## Shuffle

Episodes are shuffled at the start of each epoch for better training convergence.
