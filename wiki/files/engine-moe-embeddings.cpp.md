# `engines/trainer/moe/embeddings/embeddings.cpp` — Embedding Model Expert

**Path:** `engines/trainer/moe/embeddings/embeddings.cpp`

Embedding model MoE expert for text embedding and similarity tasks.

## EmbeddingExpert

```cpp
class EmbeddingExpert : public Expert {
    std::unique_ptr<Transformer> encoder;
    
    Tensor embed(const Tensor& text);
    Tensor compute_similarity(const Tensor& a, const Tensor& b);
    Tensor cluster(const std::vector<Tensor>& embeddings, int k);
};
```

## Operations

| Operation | Description |
|-----------|-------------|
| `embed(text)` | Generate text embeddings |
| `compute_similarity(a, b)` | Cosine similarity matrix |
| `cluster(embeddings, k)` | K-means clustering in embedding space |

## Use Cases

- Text similarity search
- Semantic clustering
- Retrieval-augmented generation (RAG)
- Classification via embedding comparison
