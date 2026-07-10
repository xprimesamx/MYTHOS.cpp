#include "oil/math.h"
#include "oil/tensor.h"

#include <cmath>
#include <cstring>
#include <algorithm>

// When OIL_AVX2 is defined, math_avx2.cpp provides all implementations.
// This file serves as the scalar reference fallback.
#if !defined(OIL_AVX2)

namespace oil {
namespace math {

static inline const float* rd(const Tensor& t) { return t.data<float>(); }
static inline float* wr(Tensor& t) { return t.data<float>(); }

float dot(const Tensor& a, const Tensor& b) {
    OIL_CHECK(a.numel() == b.numel(), "dot shape mismatch");
    const float* pa = rd(a);
    const float* pb = rd(b);
    int64_t n = a.numel();
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) s += pa[i] * pb[i];
    return s;
}

void axpy(float alpha, const Tensor& x, Tensor& y) {
    OIL_CHECK(x.numel() == y.numel(), "axpy shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) py[i] += alpha * px[i];
}

float norm(const Tensor& x) {
    const float* p = rd(x);
    int64_t n = x.numel();
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += (double)p[i] * p[i];
    return std::sqrt((float)s);
}

float asum(const Tensor& x) {
    const float* p = rd(x);
    int64_t n = x.numel();
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += std::abs((double)p[i]);
    return (float)s;
}

void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y) {
    OIL_CHECK(A.shape().rank == 2, "gemv A must be 2D");
    OIL_CHECK(x.shape().rank == 1, "gemv x must be 1D");
    OIL_CHECK(y.shape().rank == 1, "gemv y must be 1D");
    int64_t M = A.shape().dims[0];
    int64_t N = A.shape().dims[1];
    OIL_CHECK(x.numel() == N, "gemv x size != A cols");
    OIL_CHECK(y.numel() == M, "gemv y size != A rows");

    const float* pa = rd(A);
    const float* px = rd(x);
    float* py = wr(y);

    for (int64_t i = 0; i < M; i++) {
        float s = 0.0f;
        for (int64_t j = 0; j < N; j++) {
            s += pa[i * N + j] * px[j];
        }
        py[i] = alpha * s + beta * py[i];
    }
}

void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C) {
    OIL_CHECK(A.shape().rank == 2, "gemm A must be 2D");
    OIL_CHECK(B.shape().rank == 2, "gemm B must be 2D");
    OIL_CHECK(C.shape().rank == 2, "gemm C must be 2D");
    int64_t M = A.shape().dims[0];
    int64_t K = A.shape().dims[1];
    int64_t N = B.shape().dims[1];
    OIL_CHECK(B.shape().dims[0] == K, "gemm A cols != B rows");
    OIL_CHECK(C.shape().dims[0] == M && C.shape().dims[1] == N, "gemm C shape mismatch");

    const float* pa = rd(A);
    const float* pb = rd(B);
    float* pc = wr(C);

    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float s = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                s += pa[i * K + k] * pb[k * N + j];
            }
            pc[i * N + j] = alpha * s + beta * pc[i * N + j];
        }
    }
}

void relu(const Tensor& x, Tensor& y) {
    OIL_CHECK(x.numel() == y.numel(), "relu shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) py[i] = px[i] > 0.0f ? px[i] : 0.0f;
}

void gelu(const Tensor& x, Tensor& y) {
    OIL_CHECK(x.numel() == y.numel(), "gelu shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    const float s = 0.7071067811865475f; // 1/sqrt(2)
    for (int64_t i = 0; i < n; i++) {
        float v = px[i];
        py[i] = 0.5f * v * (1.0f + std::erf(v * s));
    }
}

void silu(const Tensor& x, Tensor& y) {
    OIL_CHECK(x.numel() == y.numel(), "silu shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) {
        float v = px[i];
        py[i] = v / (1.0f + std::exp(-v));
    }
}

void sigmoid(const Tensor& x, Tensor& y) {
    OIL_CHECK(x.numel() == y.numel(), "sigmoid shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) {
        py[i] = 1.0f / (1.0f + std::exp(-px[i]));
    }
}

void tanh_(const Tensor& x, Tensor& y) {
    OIL_CHECK(x.numel() == y.numel(), "tanh shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) py[i] = std::tanh(px[i]);
}

void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y) {
    OIL_CHECK(x.numel() == y.numel(), "layer_norm shape mismatch");
    OIL_CHECK(gamma.numel() == x.shape().dims[x.shape().rank-1], "layer_norm gamma dim mismatch");
    OIL_CHECK(beta.numel() == x.shape().dims[x.shape().rank-1], "layer_norm beta dim mismatch");

    int64_t last_dim = x.shape().dims[x.shape().rank - 1];
    int64_t outer = x.numel() / last_dim;

    const float* px = rd(x);
    const float* pg = rd(gamma);
    const float* pb = rd(beta);
    float* py = wr(y);

    for (int64_t i = 0; i < outer; i++) {
        const float* row = px + i * last_dim;
        double mu = 0.0;
        for (int64_t j = 0; j < last_dim; j++) mu += row[j];
        mu /= (double)last_dim;

        double var = 0.0;
        for (int64_t j = 0; j < last_dim; j++) {
            double d = row[j] - mu;
            var += d * d;
        }
        var /= (double)last_dim;

        float* ry = py + i * last_dim;
        double inv_std = 1.0 / std::sqrt(var + (double)eps);
        for (int64_t j = 0; j < last_dim; j++) {
            ry[j] = (float)(((double)row[j] - mu) * inv_std) * pg[j] + pb[j];
        }
    }
}

void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y) {
    OIL_CHECK(x.numel() == y.numel(), "rms_norm shape mismatch");
    OIL_CHECK(gamma.numel() == x.shape().dims[x.shape().rank-1], "rms_norm gamma dim mismatch");

    int64_t last_dim = x.shape().dims[x.shape().rank - 1];
    int64_t outer = x.numel() / last_dim;

    const float* px = rd(x);
    const float* pg = rd(gamma);
    float* py = wr(y);

    for (int64_t i = 0; i < outer; i++) {
        const float* row = px + i * last_dim;
        double ss = 0.0;
        for (int64_t j = 0; j < last_dim; j++) ss += (double)row[j] * row[j];
        ss /= (double)last_dim;

        double inv = 1.0 / std::sqrt(ss + (double)eps);
        float* ry = py + i * last_dim;
        for (int64_t j = 0; j < last_dim; j++) {
            ry[j] = (float)((double)row[j] * inv) * pg[j];
        }
    }
}

void softmax(const Tensor& x, Tensor& y, int axis) {
    OIL_CHECK(x.numel() == y.numel(), "softmax shape mismatch");

    Shape s = x.shape();
    int rank = s.rank;
    if (axis < 0) axis += rank;
    OIL_CHECK(axis >= 0 && axis < rank, "softmax axis out of range");

    int64_t outer = 1;
    for (int i = 0; i < axis; i++) outer *= s.dims[i];
    int64_t dim = s.dims[axis];
    int64_t inner = 1;
    for (int i = axis + 1; i < rank; i++) inner *= s.dims[i];

    const float* px = rd(x);
    float* py = wr(y);

    int64_t block = dim * inner;

    for (int64_t o = 0; o < outer; o++) {
        for (int64_t i = 0; i < inner; i++) {
            const float* src = px + o * block + i;
            float* dst = py + o * block + i;

            float mx = -INFINITY;
            for (int64_t d = 0; d < dim; d++) {
                mx = std::max(mx, src[d * inner]);
            }

            double sum = 0.0;
            for (int64_t d = 0; d < dim; d++) {
                float v = std::exp(src[d * inner] - mx);
                dst[d * inner] = v;
                sum += v;
            }

            double inv = 1.0 / sum;
            for (int64_t d = 0; d < dim; d++) {
                dst[d * inner] = (float)((double)dst[d * inner] * inv);
            }
        }
    }
}

void add(const Tensor& a, const Tensor& b, Tensor& c) {
    OIL_CHECK(a.numel() == b.numel() && b.numel() == c.numel(), "add shape mismatch");
    const float* pa = rd(a);
    const float* pb = rd(b);
    float* pc = wr(c);
    int64_t n = a.numel();
    for (int64_t i = 0; i < n; i++) pc[i] = pa[i] + pb[i];
}

void sub(const Tensor& a, const Tensor& b, Tensor& c) {
    OIL_CHECK(a.numel() == b.numel() && b.numel() == c.numel(), "sub shape mismatch");
    const float* pa = rd(a);
    const float* pb = rd(b);
    float* pc = wr(c);
    int64_t n = a.numel();
    for (int64_t i = 0; i < n; i++) pc[i] = pa[i] - pb[i];
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    OIL_CHECK(a.numel() == b.numel() && b.numel() == c.numel(), "mul shape mismatch");
    const float* pa = rd(a);
    const float* pb = rd(b);
    float* pc = wr(c);
    int64_t n = a.numel();
    for (int64_t i = 0; i < n; i++) pc[i] = pa[i] * pb[i];
}

void scale(float s, const Tensor& x, Tensor& y) {
    OIL_CHECK(x.numel() == y.numel(), "scale shape mismatch");
    const float* px = rd(x);
    float* py = wr(y);
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) py[i] = s * px[i];
}

float mean(const Tensor& x) {
    float s = sum(x);
    return s / (float)x.numel();
}

float sum(const Tensor& x) {
    const float* p = rd(x);
    int64_t n = x.numel();
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += p[i];
    return (float)s;
}

float max(const Tensor& x) {
    const float* p = rd(x);
    int64_t n = x.numel();
    OIL_CHECK(n > 0, "max on empty tensor");
    float m = p[0];
    for (int64_t i = 1; i < n; i++) m = std::max(m, p[i]);
    return m;
}

Tensor zeros_like(const Tensor& x) {
    Tensor t(x.shape(), x.dtype());
    t.zero_();
    return t;
}

Tensor ones_like(const Tensor& x) {
    Tensor t(x.shape(), x.dtype());
    t.fill(1.0f);
    return t;
}

} // namespace math
} // namespace oil

#endif // !OIL_AVX2
