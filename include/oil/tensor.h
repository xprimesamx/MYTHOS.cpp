#pragma once

#include "oil/types.h"
#include "oil/memory.h"

#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <algorithm>

namespace oil {

class Tensor {
public:
    Tensor();
    explicit Tensor(Shape shape, DType dtype = DType::F32);
    Tensor(Shape shape, std::shared_ptr<Buffer> buffer, DType dtype = DType::F32);

    Tensor(const Tensor&);
    Tensor& operator=(const Tensor&);
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;
    ~Tensor();

    const Shape& shape() const { return shape_; }
    int64_t dim(int i) const { return shape_.dims[i]; }
    int rank() const { return shape_.rank; }
    int64_t numel() const { return shape_.numel(); }
    DType dtype() const { return dtype_; }
    void* data() { return buffer_ ? (char*)buffer_->data() + offset_ : nullptr; }
    const void* data() const { return buffer_ ? (const char*)buffer_->data() + offset_ : nullptr; }
    template<typename T> T* data() { return static_cast<T*>(data()); }
    template<typename T> const T* data() const { return static_cast<const T*>(data()); }
    std::shared_ptr<Buffer> buffer() const { return buffer_; }
    size_t size_bytes() const { return numel() * dtype_size(dtype_); }

    Tensor view(const Shape& new_shape) const;
    Tensor slice(int dim, int64_t start, int64_t end) const;
    Tensor reshape(const Shape& new_shape) const;
    Tensor transpose(int dim1, int dim2) const;

    void fill(float val);
    void copy_from(const Tensor& src);
    void copy_to(Tensor& dst) const;
    Tensor clone() const;
    void zero_();

    static Tensor zeros(const Shape& shape);
    static Tensor ones(const Shape& shape);
    static Tensor arange(int64_t n);

    bool requires_grad() const { return requires_grad_; }
    void requires_grad(bool req) { requires_grad_ = req; }
    Tensor& grad() const { return *grad_; }
    bool has_grad() const { return grad_ != nullptr; }
    void set_grad(const Tensor& g) { delete grad_; grad_ = new Tensor(g); }
    void zero_grad() { if (grad_) grad_->zero_(); }

    size_t serialized_size() const;
    size_t serialize(uint8_t* dst) const;
    static Tensor deserialize(const uint8_t* src, size_t& offset);

    float& at(const std::initializer_list<int64_t>& indices);
    const float& at(const std::initializer_list<int64_t>& indices) const;

    std::string to_string() const;

private:
    Shape shape_;
    DType dtype_ = DType::F32;
    std::shared_ptr<Buffer> buffer_;
    bool requires_grad_ = false;
    Tensor* grad_ = nullptr;
    size_t offset_ = 0;
    std::vector<int64_t> strides_;

    void compute_strides();
    int64_t offset_to_flat(const std::initializer_list<int64_t>& indices) const;
    bool is_contiguous() const;
};

} // namespace oil
