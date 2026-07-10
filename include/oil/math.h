#pragma once

#include "oil/tensor.h"

namespace oil {
namespace math {

float dot(const Tensor& a, const Tensor& b);
void axpy(float alpha, const Tensor& x, Tensor& y);
float norm(const Tensor& x);
float asum(const Tensor& x);

void gemv(float alpha, const Tensor& A, const Tensor& x, float beta, Tensor& y);
void gemm(float alpha, const Tensor& A, const Tensor& B, float beta, Tensor& C);

void relu(const Tensor& x, Tensor& y);
void gelu(const Tensor& x, Tensor& y);
void silu(const Tensor& x, Tensor& y);
void sigmoid(const Tensor& x, Tensor& y);
void tanh_(const Tensor& x, Tensor& y);

void layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps, Tensor& y);
void rms_norm(const Tensor& x, const Tensor& gamma, float eps, Tensor& y);

void softmax(const Tensor& x, Tensor& y, int axis = -1);

void add(const Tensor& a, const Tensor& b, Tensor& c);
void sub(const Tensor& a, const Tensor& b, Tensor& c);
void mul(const Tensor& a, const Tensor& b, Tensor& c);
void scale(float s, const Tensor& x, Tensor& y);

float mean(const Tensor& x);
float sum(const Tensor& x);
float max(const Tensor& x);

Tensor zeros_like(const Tensor& x);
Tensor ones_like(const Tensor& x);

} // namespace math
} // namespace oil
