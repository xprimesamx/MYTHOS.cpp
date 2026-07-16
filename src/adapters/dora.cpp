#include "oil/adapters.h"

#include <cstring>
#include <cmath>
#include <stdexcept>

namespace mythos {
namespace adapters {

DoRAAdapter::DoRAAdapter(int in_dim, int out_dim, int rank, float alpha)
    : in_dim_(in_dim), out_dim_(out_dim), rank_(rank), alpha_(alpha),
      magnitude_(oil::Shape{out_dim, 1}, oil::DType::F32),
      direction_(oil::Shape{out_dim, in_dim}, oil::DType::F32),
      A_(oil::Shape{rank, in_dim}, oil::DType::F32),
      B_(oil::Shape{out_dim, rank}, oil::DType::F32) {
    if (rank <= 0) throw std::runtime_error("DoRAAdapter: rank must be positive");
    magnitude_.fill(1.0f);
    direction_.zero_();
    oil::RNG rng(42);
    float scale = 1.0f / std::sqrt((float)in_dim);
    float* ad = A_.data<float>();
    for (int64_t i = 0; i < A_.numel(); i++) ad[i] = rng.uniform() * scale;
    B_.zero_();
}

void DoRAAdapter::decompose(const Tensor& W) {
    if (W.shape() != oil::Shape{out_dim_, in_dim_})
        throw std::runtime_error("DoRAAdapter::decompose: shape mismatch");
    const float* wd = W.data<float>();
    float* md = magnitude_.data<float>();
    float* dd = direction_.data<float>();
    for (int r = 0; r < out_dim_; ++r) {
        float norm = 0;
        for (int c = 0; c < in_dim_; ++c)
            norm += wd[r * in_dim_ + c] * wd[r * in_dim_ + c];
        norm = std::sqrt(norm + 1e-10f);
        md[r] = norm;
        for (int c = 0; c < in_dim_; ++c)
            dd[r * in_dim_ + c] = wd[r * in_dim_ + c] / norm;
    }
}

Tensor DoRAAdapter::forward(const Tensor& x) const {
    if (x.dim(x.rank() - 1) != in_dim_)
        throw std::runtime_error("DoRAAdapter::forward: input dim mismatch");

    Tensor lora_out = oil::math::gemm(B_, oil::math::gemm(A_, x));
    float* ld = lora_out.data<float>();
    const float* md = magnitude_.data<float>();
    Tensor dir_out = oil::math::gemm(direction_, x);
    float* dd = dir_out.data<float>();
    for (int64_t i = 0; i < dir_out.numel(); ++i) {
        int row = (int)(i / x.dim(x.rank() - 1));
        dd[i] *= md[row % out_dim_];
        dd[i] += (alpha_ / (float)rank_) * ld[i];
    }
    return dir_out;
}

void DoRAAdapter::save(const std::string& oil_path) const {
    oil::OILWriter writer(oil_path);
    writer.write_tensor("dora_magnitude", magnitude_);
    writer.write_tensor("dora_direction", direction_);
    writer.write_tensor("dora_A", A_);
    writer.write_tensor("dora_B", B_);
    writer.close();
}

void DoRAAdapter::load(const std::string& oil_path) {
    oil::OILReader reader(oil_path);
    magnitude_ = reader.read_tensor("dora_magnitude");
    direction_ = reader.read_tensor("dora_direction");
    A_ = reader.read_tensor("dora_A");
    B_ = reader.read_tensor("dora_B");
}

} // namespace adapters
} // namespace mythos
