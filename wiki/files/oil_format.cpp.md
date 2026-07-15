# `oil_format.cpp` — OIL Format I/O Implementation

**Path:** `src/oil_format.cpp`

Reads and writes the OIL binary format for model persistence.

## Write Pipeline

```
OILWriter:
1. Write magic bytes "OIL\0" (4 bytes)
2. Write header: version, tensor count, offsets
3. Write model config (JSON as string)
4. For each tensor:
   a. Write metadata (name, shape, dtype, offset)
   b. Write raw tensor data
5. Write optional tokenizer data
6. Finalize: update offsets in header
```

## Read Pipeline

```
OILReader:
1. Verify magic bytes "OIL\0"
2. Read header → version, offsets
3. Read config from metadata_offset
4. Build tensor index from metadata
5. Lazy tensor loading: read on demand
```

## File Format

```
┌──────────────────────────┐
│ OIL Header (64 bytes)    │ ← magic, version, offsets
├──────────────────────────┤
│ Config (JSON string)     │ ← TransformerConfig
├──────────────────────────┤
│ Tensor Metadata Table    │ ← name, shape, dtype, file_offset
├──────────────────────────┤
│ Weight Data Section      │ ← raw tensor bytes
├──────────────────────────┤
│ Tokenizer Section (opt)  │ ← vocab + merges
└──────────────────────────┘
```

## Version Compatibility

- Version 1: Initial format (current)
- Forward compatibility via version field
- Backward compatibility through format negotiation
