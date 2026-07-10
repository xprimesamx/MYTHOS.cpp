#include "oil/tensor.h"
#include "oil/types.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

int main() {
    // Test Shape
    {
        oil::Shape s1(2, 3);
        assert(s1.numel() == 6);
        assert(s1.rank == 2);
        assert(s1.dims[0] == 2);
        assert(s1.dims[1] == 3);

        oil::Shape s2(std::initializer_list<int64_t>{4, 5, 6});
        assert(s2.rank == 3);
        assert(s2.numel() == 120);

        oil::Shape s3;
        assert(s3.rank == 0);
        assert(s3.numel() == 1);

        assert(s1 == oil::Shape(2, 3));
        assert(!(s1 == s2));
        assert(s1 != s2);
    }

    // Test Tensor zeros
    {
        auto t = oil::Tensor::zeros({2, 3});
        assert(t.numel() == 6);
        assert(t.dtype() == oil::DType::F32);
        assert(t.shape().rank == 2);
        assert(t.shape().dims[0] == 2);
        assert(t.shape().dims[1] == 3);
        for (int64_t i = 0; i < t.numel(); i++)
            assert(t.data<float>()[i] == 0.0f);
    }

    // Test Tensor ones
    {
        auto t = oil::Tensor::ones({4});
        assert(t.numel() == 4);
        for (int64_t i = 0; i < t.numel(); i++)
            assert(t.data<float>()[i] == 1.0f);
    }

    // Test arange
    {
        auto t = oil::Tensor::arange(5);
        assert(t.numel() == 5);
        for (int64_t i = 0; i < 5; i++)
            assert(t.data<float>()[i] == (float)i);
    }

    // Test fill
    {
        auto t = oil::Tensor::zeros({2, 2});
        t.fill(3.14f);
        for (int64_t i = 0; i < t.numel(); i++)
            assert(std::abs(t.data<float>()[i] - 3.14f) < 1e-6f);
    }

    // Test at
    {
        auto t = oil::Tensor::arange(6).reshape({2, 3});
        assert(t.at({0, 0}) == 0.0f);
        assert(t.at({0, 1}) == 1.0f);
        assert(t.at({1, 2}) == 5.0f);
        t.at({1, 1}) = 99.0f;
        assert(t.at({1, 1}) == 99.0f);
    }

    // Test view
    {
        auto t = oil::Tensor::arange(6);
        auto v = t.view({2, 3});
        assert(v.shape().rank == 2);
        assert(v.shape().dims[0] == 2);
        assert(v.shape().dims[1] == 3);
        assert(v.at({0, 0}) == 0.0f);
        assert(v.at({1, 2}) == 5.0f);
        v.at({1, 1}) = 42.0f;
        assert(t.data<float>()[4] == 42.0f);
    }

    // Test reshape
    {
        auto t = oil::Tensor::arange(12);
        auto r = t.reshape({3, 4});
        assert(r.shape().dims[0] == 3);
        assert(r.shape().dims[1] == 4);
        assert(r.numel() == 12);
    }

    // Test slice
    {
        auto t = oil::Tensor::arange(10);
        auto s = t.slice(0, 2, 5);
        assert(s.numel() == 3);
        assert(s.data<float>()[0] == 2.0f);
        assert(s.data<float>()[1] == 3.0f);
        assert(s.data<float>()[2] == 4.0f);
    }

    // Test transpose
    {
        auto t = oil::Tensor::arange(6).reshape({2, 3});
        auto tp = t.transpose(0, 1);
        assert(tp.shape().dims[0] == 3);
        assert(tp.shape().dims[1] == 2);
        assert(tp.at({0, 0}) == t.at({0, 0}));
        assert(tp.at({1, 0}) == t.at({0, 1}));
        assert(tp.at({2, 1}) == t.at({1, 2}));
    }

    // Test clone
    {
        auto t = oil::Tensor::arange(10);
        auto c = t.clone();
        assert(c.numel() == t.numel());
        for (int64_t i = 0; i < t.numel(); i++)
            assert(c.data<float>()[i] == t.data<float>()[i]);
        c.data<float>()[0] = 999.0f;
        assert(t.data<float>()[0] == 0.0f);
    }

    // Test copy_from / copy_to
    {
        auto a = oil::Tensor::arange(5);
        auto b = oil::Tensor::zeros({5});
        a.copy_to(b);
        for (int64_t i = 0; i < 5; i++)
            assert(b.data<float>()[i] == (float)i);
        b.data<float>()[0] = 100.0f;
        a.copy_from(b);
        assert(a.data<float>()[0] == 100.0f);
    }

    // Test serialization round-trip
    {
        auto t = oil::Tensor::arange(42).reshape({6, 7});
        std::vector<uint8_t> buf(t.serialized_size());
        t.serialize(buf.data());
        size_t offset = 0;
        auto t2 = oil::Tensor::deserialize(buf.data(), offset);
        assert(t2.numel() == t.numel());
        assert(t2.shape() == t.shape());
        assert(t2.dtype() == t.dtype());
        for (int64_t i = 0; i < t.numel(); i++)
            assert(std::abs(t2.data<float>()[i] - t.data<float>()[i]) < 1e-6f);
    }

    // Test zero_
    {
        auto t = oil::Tensor::ones({10});
        t.zero_();
        for (int64_t i = 0; i < t.numel(); i++)
            assert(t.data<float>()[i] == 0.0f);
    }

    // Test to_string
    {
        auto t = oil::Tensor::arange(4);
        std::string s = t.to_string();
        assert(!s.empty());
        assert(s.find("Tensor") != std::string::npos);
    }

    std::cout << "All tensor tests passed!" << std::endl;
    return 0;
}
