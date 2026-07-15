# `codebook.cpp` — Codebook Training Implementation

**Path:** `src/codebook.cpp`

K-means codebook training for vector quantization.

## K-Means Algorithm

```
1. Initialize k centroids (k-means++ or random)
2. Repeat until converged (max 100 iterations):
   a. Assign each data point to nearest centroid
   b. Update centroids = mean of assigned points
   c. Check convergence (centroid movement < eps)
3. Return centroids
```

## Functions

| Function | Description |
|----------|-------------|
| `train_kmeans(data, k)` | Standard k-means clustering |
| `train_kmeans_pp(data, k)` | K-means++ initialization |
| `compute_distances(data, centroids)` | Pairwise distance matrix |
| `assign_clusters(distances)` | Nearest centroid assignment |
| `update_centroids(data, assignments, k)` | Recompute centroids |

## Distance Computation

Uses squared Euclidean distance with AVX2 optimization for batch computation.
