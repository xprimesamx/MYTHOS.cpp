#include "oil/adapters.h"

#include <cstring>
#include <stdexcept>

namespace mythos {
namespace adapters {

LoRAAdapter::LoRAAdapter(int in_dim, int out_dim, int rank, float alpha)
    : in_dim_(in_dim), out_dim_(out_dim), rank_(rank), alpha_(alpha),
      base_(oil::Shape{out_dim, in_dim}, oil::DType::F32),
      A_(oil::Shape{rank, in_dim}, oil::DType::F32),
      B_(oil::Shape{out_dim, rank}, oil::DType::F32) {
    if (rank <= 0) throw std::runtime_error("LoRAAdapter: rank must be positive");
    if (in_dim <= 0 || out_dim <= 0) throw std::runtime_error("LoRAAdapter: dims must be positive");
    base_.zero_();
    oil::RNG rng(42);
    float scale = 1.0f / std::sqrt((float)in_dim);
    float* ad = A_.data<float>();
    for (int64_t i = 0; i < A_.numel(); i++) ad[i] = rng.uniform() * scale;
    B_.zero_();
}

void LoRAAdapter::set_base(const Tensor& W) {
    if (W.shape() != oil::Shape{out_dim_, in_dim_})
        throw std::runtime_error("LoRAAdapter::set_base: shape mismatch");
    base_.copy_from(W);
}

Tensor LoRAAdapter::forward(const Tensor& x) const {
    if (x.shape().rank != 1 || x.dim(0) != in_dim_)
        throw std::runtime_error("LoRAAdapter::forward: input shape mismatch");
    Tensor out(oil::Shape{out_dim_}, oil::DType::F32);
    out.zero_();
    oil::math::gemv(1.0f, base_, x, 0.0f, out);
    Tensor tmp(oil::Shape{rank_}, oil::DType::F32);
    tmp.zero_();
    oil::math::gemv(1.0f, A_, x, 0.0f, tmp);
    Tensor delta(oil::Shape{out_dim_}, oil::DType::F32);
    delta.zero_();
    oil::math::gemv(1.0f, B_, tmp, 0.0f, delta);
    float s = alpha_ / (float)rank_;
    float* od = out.data<float>();
    const float* dd = delta.data<float>();
    for (int i = 0; i < out_dim_; i++) od[i] += s * dd[i];
    return out;
}

void LoRAAdapter::save(const std::string& oil_path) const {
    oil::OILWriter w(oil_path);
    oil::OILHeader hdr;
    std::memcpy(hdr.magic, "OIL1", 4);
    hdr.version = 1; hdr.flags = 0; hdr.config_size = 0;
    w.write_header(hdr, nullptr);
    oil::FormatBlockEntry fe{0, (uint8_t)oil::Format::FP32, 0};
    w.write_format_table({fe});
    std::vector<oil::TensorEntry> te(3);
    std::vector<std::string> names = {"lora.base", "lora.A", "lora.B"};
    for (int i = 0; i < 3; i++) { te[i].name_len = (uint16_t)names[i].size(); te[i].block_start = (uint32_t)i; te[i].num_blocks = 1; }
    w.write_tensor_table(te, names);
    const Tensor* mats[3] = {&base_, &A_, &B_};
    for (int i = 0; i < 3; i++) {
        oil::BlockData b;
        b.format = oil::Format::FP32;
        b.num_weights = (uint32_t)mats[i]->numel();
        b.indices.resize(b.num_weights * 4);
        std::memcpy(b.indices.data(), mats[i]->data(), b.indices.size());
        w.write_block(b);
    }
    w.close();
}

void LoRAAdapter::load(const std::string& oil_path) {
    oil::OILReader r(oil_path);
    if (!r.valid()) throw std::runtime_error("LoRAAdapter::load: invalid OIL file");
    auto names = r.tensor_names();
    for (const auto& nm : names) {
        Tensor t = r.read_tensor(nm);
        if (nm == "lora.base") base_.copy_from(t.reshape(oil::Shape{out_dim_, in_dim_}));
        else if (nm == "lora.A") A_.copy_from(t.reshape(oil::Shape{rank_, in_dim_}));
        else if (nm == "lora.B") B_.copy_from(t.reshape(oil::Shape{out_dim_, rank_}));
    }
}

} // namespace adapters
} // namespace mythos
