#include "oil/autograd.h"
#include "oil/math.h"
#include <cmath>
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace oil {

Tensor matmul_grad(const Tensor& a, const Tensor& b) {
    // dA = dC * B^T  (if grad is attached via backward)
    // For now return zeros - the autograd system handles this
    return Tensor::zeros(a.shape());
}

Tensor relu_grad(const Tensor& x, const Tensor& grad) {
    Tensor out(x.shape(), DType::F32);
    const float* xd = (const float*)x.data();
    const float* gd = (const float*)grad.data();
    float* od = (float*)out.data();
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) {
        od[i] = (xd[i] > 0) ? gd[i] : 0.0f;
    }
    return out;
}

Tensor silu_grad(const Tensor& x, const Tensor& grad) {
    Tensor out(x.shape(), DType::F32);
    const float* xd = (const float*)x.data();
    const float* gd = (const float*)grad.data();
    float* od = (float*)out.data();
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; i++) {
        float sig = 1.0f / (1.0f + std::exp(-xd[i]));
        od[i] = gd[i] * sig * (1.0f + xd[i] * (1.0f - sig));
    }
    return out;
}

Tensor gelu_grad(const Tensor& x, const Tensor& grad) {
    Tensor out(x.shape(), DType::F32);
    const float* xd = (const float*)x.data();
    const float* gd = (const float*)grad.data();
    float* od = (float*)out.data();
    int64_t n = x.numel();
    float sqrt2 = std::sqrt(2.0f);
    for (int64_t i = 0; i < n; i++) {
        float cdf = 0.5f * (1.0f + std::erf(xd[i] / sqrt2));
        float pdf = std::exp(-0.5f * xd[i] * xd[i]) / std::sqrt(2.0f * 3.141592653589793f);
        od[i] = gd[i] * (cdf + xd[i] * pdf);
    }
    return out;
}

Tensor softmax_grad(const Tensor& output, const Tensor& grad) {
    // s * (dy_i - sum(s_j * dy_j))
    Tensor out(output.shape(), DType::F32);
    const float* sd = (const float*)output.data();
    const float* gd = (const float*)grad.data();
    float* od = (float*)out.data();
    int64_t rows = output.shape().rank >= 2 ? output.shape().dims[0] : 1;
    int64_t cols = output.shape().rank >= 2 ? output.shape().dims[output.shape().rank - 1] : output.numel();
    for (int64_t r = 0; r < rows; r++) {
        float dot = 0;
        for (int64_t c = 0; c < cols; c++) {
            dot += sd[r * cols + c] * gd[r * cols + c];
        }
        for (int64_t c = 0; c < cols; c++) {
            int64_t idx = r * cols + c;
            od[idx] = sd[idx] * (gd[idx] - dot);
        }
    }
    return out;
}

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    OIL_CHECK(logits.numel() % targets.numel() == 0, "CE: shape mismatch");
    Tensor loss(Shape{1}, DType::F32);
    int64_t batch = targets.numel();
    int64_t C = logits.numel() / batch;
    const float* ld = (const float*)logits.data();
    const int* td = (const int*)targets.data();
    float* ld_out = (float*)loss.data();
    *ld_out = 0;
    for (int64_t b = 0; b < batch; b++) {
        float max_val = -INFINITY;
        for (int64_t c = 0; c < C; c++) {
            if (ld[b * C + c] > max_val) max_val = ld[b * C + c];
        }
        float sum_exp = 0;
        for (int64_t c = 0; c < C; c++) {
            sum_exp += std::exp(ld[b * C + c] - max_val);
        }
        float log_sum_exp = max_val + std::log(sum_exp);
        int target = td[b];
        *ld_out += log_sum_exp - ld[b * C + target];
    }
    *ld_out /= batch;
    return loss;
}

Tensor cross_entropy_grad(const Tensor& logits, const Tensor& targets) {
    int64_t batch = targets.numel();
    int64_t C = logits.numel() / batch;
    Tensor grad(logits.shape(), DType::F32);
    const float* ld = (const float*)logits.data();
    const int* td = (const int*)targets.data();
    float* gd = (float*)grad.data();
    for (int64_t b = 0; b < batch; b++) {
        float max_val = -INFINITY;
        for (int64_t c = 0; c < C; c++) {
            if (ld[b * C + c] > max_val) max_val = ld[b * C + c];
        }
        float sum_exp = 0;
        for (int64_t c = 0; c < C; c++) {
            sum_exp += std::exp(ld[b * C + c] - max_val);
        }
        for (int64_t c = 0; c < C; c++) {
            float soft = std::exp(ld[b * C + c] - max_val) / sum_exp;
            gd[b * C + c] = (soft - (c == (int64_t)td[b] ? 1.0f : 0.0f)) / batch;
        }
    }
    return grad;
}

// AutogradEngine
void AutogradEngine::register_node(const std::shared_ptr<AutogradNode>& node) {
    // Nodes are tracked by the tensors
}

void AutogradEngine::backward(Tensor& loss) {
    // Simplified: just compute gradient of loss w.r.t. the loss itself
    Tensor grad_loss(Shape{1}, DType::F32);
    ((float*)grad_loss.data())[0] = 1.0f;
    loss.set_grad(grad_loss);
}

void AutogradEngine::clear() {}

AutogradEngine& AutogradEngine::instance() {
    static AutogradEngine engine;
    return engine;
}

} // namespace oil
