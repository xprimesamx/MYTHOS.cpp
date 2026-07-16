#pragma once

#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/oil_format.h"
#include "oil/math.h"

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace mythos {
namespace adapters {

struct TensorRecord {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
};

struct GGUFMeta {
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    std::string arch;
};

class LoRAAdapter {
public:
    LoRAAdapter(int in_dim, int out_dim, int rank, float alpha);
    Tensor forward(const Tensor& x) const;
    void set_base(const Tensor& W);
    const Tensor& base() const { return base_; }
    const Tensor& A() const { return A_; }
    const Tensor& B() const { return B_; }
    int rank() const { return rank_; }
    float alpha() const { return alpha_; }
    void save(const std::string& oil_path) const;
    void load(const std::string& oil_path);
private:
    int in_dim_, out_dim_, rank_;
    float alpha_;
    Tensor base_;
    Tensor A_;
    Tensor B_;
};

class QLoRAAdapter {
public:
    QLoRAAdapter(int in_dim, int out_dim, int rank, float alpha);
    void quantize_base(const Tensor& W);
    Tensor forward(const Tensor& x) const;
    Tensor dequantize_base() const;
    int rank() const { return rank_; }
    float alpha() const { return alpha_; }
    void save(const std::string& oil_path) const;
    void load(const std::string& oil_path);
private:
    int in_dim_, out_dim_, rank_;
    float alpha_;
    std::vector<float> codebook_;
    std::vector<uint8_t> indices_;
    int num_weights_;
    Tensor A_;
    Tensor B_;
    void dequant_row(int row, float* out) const;
};

class DoRAAdapter {
public:
    DoRAAdapter(int in_dim, int out_dim, int rank, float alpha);
    void decompose(const Tensor& W);
    Tensor forward(const Tensor& x) const;
    const Tensor& magnitude() const { return magnitude_; }
    const Tensor& direction() const { return direction_; }
    int rank() const { return rank_; }
    float alpha() const { return alpha_; }
    void save(const std::string& oil_path) const;
    void load(const std::string& oil_path);
private:
    int in_dim_, out_dim_, rank_;
    float alpha_;
    Tensor magnitude_;
    Tensor direction_;
    Tensor A_;
    Tensor B_;
};

class GGUFImporter {
public:
    GGUFImporter();
    bool read(const std::string& path);
    bool write_oil(const std::string& out_path) const;
    const GGUFMeta& meta() const { return meta_; }
    const std::vector<TensorRecord>& tensors() const { return tensors_; }
    static bool import_gguf(const std::string& in_path, const std::string& out_path);
private:
    GGUFMeta meta_;
    std::vector<TensorRecord> tensors_;
};

class SafetensorsImporter {
public:
    SafetensorsImporter();
    bool read(const std::string& path);
    bool write_oil(const std::string& out_path) const;
    const std::vector<TensorRecord>& tensors() const { return tensors_; }
    static bool import_safetensors(const std::string& in_path, const std::string& out_path);
private:
    std::vector<TensorRecord> tensors_;
};

bool is_oil_path(const std::string& path);
std::unique_ptr<oil::OILReader> oil_load(const std::string& path);

} // namespace adapters
} // namespace mythos
