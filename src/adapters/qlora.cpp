#include "oil/adapters.h"

#include <cstring>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace mythos {
namespace adapters {

QLoRAAdapter::QLoRAAdapter(int in_dim, int out_dim, int rank, float alpha)
    : in_dim_(in_dim), out_dim_(out_dim), rank_(rank), alpha_(alpha), num_weights_(0),
      A_(oil::Shape{rank, in_dim}, oil::DType::F32),
      B_(oil::Shape{out_dim, rank}, oil::DType::F32) {
    if (rank <= 0) throw std::runtime_error("QLoRAAdapter: rank must be positive");
    oil::RNG rng(7);
    float scale = 1.0f / std::sqrt((float)in_dim);
    float* ad = A_.data<float>();
    for (int64_t i = 0; i < A_.numel(); i++) ad[i] = rng.uniform() * scale;
    B_.zero_();
}

static const float kNF4Codebook[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477143037319183f, -0.09110999757051468f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

void QLoRAAdapter::quantize_base(const Tensor& W) {
    if (W.shape() != oil::Shape{out_dim_, in_dim_})
        throw std::runtime_error("QLoRAAdapter::quantize_base: shape mismatch");
    num_weights_ = out_dim_ * in_dim_;
    const float* wd = W.data<float>();
    float wmin = wd[0], wmax = wd[0];
    for (int i = 1; i < num_weights_; i++) {
        wmin = std::min(wmin, wd[i]);
        wmax = std::max(wmax, wd[i]);
    }
    float range = wmax - wmin;
    float scale = (range > 1e-12f) ? range / 2.0f : 1.0f;
    float zero = (wmax + wmin) * 0.5f;
    codebook_.resize(16);
    for (int k = 0; k < 16; k++) codebook_[k] = kNF4Codebook[k] * scale + zero;
    indices_.assign((num_weights_ + 1) / 2, 0);
    for (int i = 0; i < num_weights_; i++) {
        float best = codebook_[0];
        uint8_t bi = 0;
        for (int k = 1; k < 16; k++) {
            if (std::fabs(wd[i] - codebook_[k]) < std::fabs(wd[i] - best)) { best = codebook_[k]; bi = (uint8_t)k; }
        }
        if (i % 2 == 0) indices_[i / 2] = bi & 0x0F;
        else            indices_[i / 2] |= (bi << 4);
    }
}

void QLoRAAdapter::dequant_row(int row, float* out) const {
    int base = row * in_dim_;
    for (int j = 0; j < in_dim_; j++) {
        int idx = base + j;
        uint8_t q = (idx % 2 == 0) ? (indices_[idx / 2] & 0x0F) : ((indices_[idx / 2] >> 4) & 0x0F);
        out[j] = codebook_[q];
    }
}

Tensor QLoRAAdapter::dequantize_base() const {
    Tensor W(oil::Shape{out_dim_, in_dim_}, oil::DType::F32);
    float* wd = W.data<float>();
    for (int r = 0; r < out_dim_; r++) dequant_row(r, wd + r * in_dim_);
    return W;
}

Tensor QLoRAAdapter::forward(const Tensor& x) const {
    if (x.shape().rank != 1 || x.dim(0) != in_dim_)
        throw std::runtime_error("QLoRAAdapter::forward: input shape mismatch");
    Tensor out(oil::Shape{out_dim_}, oil::DType::F32);
    float* od = out.data<float>();
    const float* xd = x.data<float>();
    std::vector<float> row(in_dim_);
    for (int r = 0; r < out_dim_; r++) {
        dequant_row(r, row.data());
        float acc = 0.0f;
        for (int j = 0; j < in_dim_; j++) acc += row[j] * xd[j];
        od[r] = acc;
    }
    Tensor tmp(oil::Shape{rank_}, oil::DType::F32);
    tmp.zero_();
    oil::math::gemv(1.0f, A_, x, 0.0f, tmp);
    Tensor delta(oil::Shape{out_dim_}, oil::DType::F32);
    delta.zero_();
    oil::math::gemv(1.0f, B_, tmp, 0.0f, delta);
    float s = alpha_ / (float)rank_;
    const float* dd = delta.data<float>();
    for (int i = 0; i < out_dim_; i++) od[i] += s * dd[i];
    return out;
}

void QLoRAAdapter::save(const std::string& oil_path) const {
    oil::OILWriter w(oil_path);
    oil::OILHeader hdr;
    std::memcpy(hdr.magic, "OIL1", 4);
    hdr.version = 1; hdr.flags = 0; hdr.config_size = 0;
    w.write_header(hdr, nullptr);
    oil::FormatBlockEntry fe{0, (uint8_t)oil::Format::FP32, 0};
    w.write_format_table({fe});
    std::vector<oil::TensorEntry> te(4);
    std::vector<std::string> names = {"qlora.A", "qlora.B", "qlora.codebook", "qlora.indices"};
    for (int i = 0; i < 4; i++) { te[i].name_len = (uint16_t)names[i].size(); te[i].block_start = (uint32_t)i; te[i].num_blocks = 1; }
    w.write_tensor_table(te, names);
    auto writef = [&](const float* p, int64_t n) {
        oil::BlockData b; b.format = oil::Format::FP32; b.num_weights = (uint32_t)n;
        b.indices.resize(n * 4); std::memcpy(b.indices.data(), p, n * 4); w.write_block(b);
    };
    writef(A_.data<float>(), A_.numel());
    writef(B_.data<float>(), B_.numel());
    writef(codebook_.data(), (int64_t)codebook_.size());
    oil::BlockData bi; bi.format = oil::Format::FP32; bi.num_weights = (uint32_t)indices_.size();
    bi.indices.assign(indices_.begin(), indices_.end()); w.write_block(bi);
    w.close();
}

void QLoRAAdapter::load(const std::string& oil_path) {
    oil::OILReader r(oil_path);
    if (!r.valid()) throw std::runtime_error("QLoRAAdapter::load: invalid OIL file");
    for (const auto& nm : r.tensor_names()) {
        Tensor t = r.read_tensor(nm);
        if (nm == "qlora.A") A_.copy_from(t.reshape(oil::Shape{rank_, in_dim_}));
        else if (nm == "qlora.B") B_.copy_from(t.reshape(oil::Shape{out_dim_, rank_}));
        else if (nm == "qlora.codebook") {
            codebook_.resize(t.numel());
            std::memcpy(codebook_.data(), t.data(), t.numel() * 4);
        } else if (nm == "qlora.indices") {
            indices_.assign((uint8_t*)t.data(), (uint8_t*)t.data() + t.numel());
            num_weights_ = out_dim_ * in_dim_;
        }
    }
}

} // namespace adapters
} // namespace mythos
