#include "oil/tensor.h"
#include "oil/memory.h"
#include "oil/types.h"
#include <cstring>
#include <sstream>
#include <iomanip>

namespace oil {

Tensor::Tensor()
    : shape_(), dtype_(DType::F32), requires_grad_(false),
      grad_(nullptr), offset_(0)
{}

Tensor::Tensor(Shape shape, DType dtype)
    : shape_(shape), dtype_(dtype), requires_grad_(false),
      grad_(nullptr), offset_(0)
{
    size_t bytes = (size_t)shape_.numel() * ::oil::dtype_size(dtype);
    buffer_ = std::make_shared<Buffer>(bytes, 64);
    if (buffer_->data()) memset(buffer_->data(), 0, bytes);
    compute_strides();
}

Tensor::Tensor(Shape shape, std::shared_ptr<Buffer> buffer, DType dtype)
    : shape_(shape), dtype_(dtype), buffer_(buffer),
      requires_grad_(false), grad_(nullptr), offset_(0)
{
    compute_strides();
}

Tensor::~Tensor() {
    delete grad_;
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), buffer_(other.buffer_),
      offset_(other.offset_), strides_(other.strides_),
      requires_grad_(other.requires_grad_)
{
    if (other.grad_) {
        grad_ = new Tensor(*other.grad_);
    } else {
        grad_ = nullptr;
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        buffer_ = other.buffer_;
        offset_ = other.offset_;
        strides_ = other.strides_;
        requires_grad_ = other.requires_grad_;
        delete grad_;
        if (other.grad_) {
            grad_ = new Tensor(*other.grad_);
        } else {
            grad_ = nullptr;
        }
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(other.shape_), dtype_(other.dtype_),
      buffer_(std::move(other.buffer_)),
      offset_(other.offset_), strides_(std::move(other.strides_)),
      requires_grad_(other.requires_grad_), grad_(other.grad_)
{
    other.grad_ = nullptr;
    other.offset_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        buffer_ = std::move(other.buffer_);
        offset_ = other.offset_;
        strides_ = std::move(other.strides_);
        requires_grad_ = other.requires_grad_;
        delete grad_;
        grad_ = other.grad_;
        other.grad_ = nullptr;
        other.offset_ = 0;
    }
    return *this;
}

void Tensor::compute_strides() {
    strides_.resize(shape_.rank);
    int64_t s = 1;
    for (int i = shape_.rank - 1; i >= 0; i--) {
        strides_[i] = s;
        s *= shape_.dims[i];
    }
}

Tensor Tensor::view(const Shape& new_shape) const {
    Tensor t;
    t.shape_ = new_shape;
    t.dtype_ = dtype_;
    t.buffer_ = buffer_;
    t.offset_ = offset_;
    t.requires_grad_ = requires_grad_;
    t.compute_strides();
    return t;
}

Tensor Tensor::slice(int dim, int64_t start, int64_t end) const {
    OIL_CHECK(dim < shape_.rank, "slice: dim out of range");
    OIL_CHECK(start >= 0 && end <= shape_.dims[dim], "slice: bounds");
    Tensor t;
    t.shape_ = shape_;
    t.shape_.dims[dim] = end - start;
    t.dtype_ = dtype_;
    t.buffer_ = buffer_;
    t.offset_ = offset_ + (size_t)(start * strides_[dim]) * ::oil::dtype_size(dtype_);
    t.requires_grad_ = requires_grad_;
    t.compute_strides();
    return t;
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    OIL_CHECK(new_shape.numel() == shape_.numel(), "reshape: size mismatch");
    return view(new_shape);
}

Tensor Tensor::transpose(int dim1, int dim2) const {
    OIL_CHECK(dim1 < shape_.rank && dim2 < shape_.rank, "transpose: dim out of range");
    int64_t n = numel();
    std::vector<int64_t> src_strides(rank());
    src_strides[rank()-1] = 1;
    for (int r = rank()-2; r >= 0; --r)
        src_strides[r] = src_strides[r+1] * shape_.dims[r+1];

    Shape dst_shape = shape_;
    std::swap(dst_shape.dims[dim1], dst_shape.dims[dim2]);
    std::vector<int> dim_map(rank());
    for (int r = 0; r < rank(); ++r) dim_map[r] = r;
    std::swap(dim_map[dim1], dim_map[dim2]);

    Tensor t(dst_shape, dtype_);
    t.requires_grad_ = requires_grad_;
    const char* src = static_cast<const char*>(data());
    char* dst = static_cast<char*>(t.data());
    int64_t elem_bytes = dtype_size(dtype_);
    for (int64_t i = 0; i < n; i++) {
        int64_t src_off = 0, rem = i;
        for (int r = rank()-1; r >= 0; --r) {
            int64_t idx = rem % dst_shape.dims[r];
            rem /= dst_shape.dims[r];
            src_off += idx * src_strides[dim_map[r]];
        }
        memcpy(dst + i * elem_bytes, src + src_off * elem_bytes, elem_bytes);
    }
    return t;
}

void Tensor::fill(float val) {
    float* d = data<float>();
    int64_t n = numel();
    for (int64_t i = 0; i < n; i++) d[i] = val;
}

void Tensor::zero_() {
    if (data()) memset(data(), 0, size_bytes());
}

void Tensor::copy_from(const Tensor& src) {
    OIL_CHECK(shape_.numel() == src.shape_.numel(), "copy_from: size mismatch");
    memcpy(data(), src.data(), size_bytes());
}

void Tensor::copy_to(Tensor& dst) const {
    dst.copy_from(*this);
}

Tensor Tensor::clone() const {
    Tensor t(shape_, dtype_);
    t.requires_grad_ = requires_grad_;
    if (data() && t.data()) memcpy(t.data(), data(), size_bytes());
    return t;
}

Tensor Tensor::zeros(const Shape& shape) {
    Tensor t(shape, DType::F32);
    t.zero_();
    return t;
}

Tensor Tensor::ones(const Shape& shape) {
    Tensor t(shape, DType::F32);
    t.fill(1.0f);
    return t;
}

Tensor Tensor::arange(int64_t n) {
    Tensor t(Shape{n}, DType::F32);
    float* d = t.data<float>();
    for (int64_t i = 0; i < n; i++) d[i] = (float)i;
    return t;
}

int64_t Tensor::offset_to_flat(const std::initializer_list<int64_t>& indices) const {
    int64_t idx = 0;
    int i = 0;
    for (auto ix : indices) {
        OIL_CHECK(ix < shape_.dims[i], "index out of bounds");
        idx += ix * strides_[i];
        i++;
    }
    return idx;
}

float& Tensor::at(const std::initializer_list<int64_t>& indices) {
    return data<float>()[offset_to_flat(indices)];
}

const float& Tensor::at(const std::initializer_list<int64_t>& indices) const {
    return data<float>()[offset_to_flat(indices)];
}

size_t Tensor::serialized_size() const {
    return sizeof(int32_t) + (size_t)shape_.rank * sizeof(int64_t) + sizeof(uint8_t) + size_bytes();
}

size_t Tensor::serialize(uint8_t* dst) const {
    uint8_t* p = dst;
    int32_t r = (int32_t)shape_.rank;
    memcpy(p, &r, sizeof(r)); p += sizeof(r);
    for (int i = 0; i < shape_.rank; i++) {
        int64_t d = shape_.dims[i];
        memcpy(p, &d, sizeof(d)); p += sizeof(d);
    }
    uint8_t dt = (uint8_t)dtype_;
    memcpy(p, &dt, sizeof(dt)); p += sizeof(dt);
    size_t bytes = size_bytes();
    if (data() && bytes) memcpy(p, data(), bytes);
    p += bytes;
    return (size_t)(p - dst);
}

Tensor Tensor::deserialize(const uint8_t* src, size_t& offset) {
    const uint8_t* p = src + offset;
    int32_t r; memcpy(&r, p, sizeof(r)); p += sizeof(r);
    Shape shape;
    shape.rank = r;
    for (int i = 0; i < r; i++) {
        int64_t d; memcpy(&d, p, sizeof(d)); p += sizeof(d);
        shape.dims[i] = d;
    }
    uint8_t dt; memcpy(&dt, p, sizeof(dt)); p += sizeof(dt);
    DType dtype = (DType)dt;
    Tensor t(shape, dtype);
    size_t bytes = t.size_bytes();
    if (bytes && t.data()) memcpy(t.data(), p, bytes);
    p += bytes;
    offset = (size_t)(p - src);
    return t;
}

bool Tensor::is_contiguous() const { return true; }

std::string Tensor::to_string() const {
    std::ostringstream os;
    os << "Tensor" << shape_.to_string() << " " << size_bytes() << "B";
    if (numel() <= 16 && dtype_ == DType::F32) {
        os << " [";
        const float* d = static_cast<const float*>(data());
        for (int64_t i = 0; i < numel(); i++) {
            if (i) os << ", ";
            os << std::fixed << std::setprecision(4) << d[i];
        }
        os << "]";
    }
    return os.str();
}

} // namespace oil
