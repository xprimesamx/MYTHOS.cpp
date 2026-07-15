# `asi.cpp` — ASI Implementation

**Path:** `src/asi.cpp`

Advanced Synthetic Intelligence module implementation: multi-step reasoning and reflection.

## Functions

| Function | Description |
|----------|-------------|
| `ASI::think()` | Generate reasoning steps |
| `ASI::reason()` | Multi-step reasoning chain |
| `ASI::learn()` | Incorporate feedback |
| `ASI::reflect()` | Self-critique outputs |

## Reasoning Process

```
ASI.think(input):
1. Generate candidate thoughts (num_candidates)
2. Score each thought
3. Select best thought
4. Repeat for reasoning_depth steps
5. Synthesize final output
```

## Reflection

Self-reflection improves output quality:
```
1. Generate initial response
2. Model critiques own response
3. Refine based on critique
4. Repeat for N iterations
```
