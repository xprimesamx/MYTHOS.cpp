#pragma once

#include "oil/tensor.h"
#include <memory>
#include <vector>
#include <unordered_map>

namespace oil {

class AutogradFunction {
public:
    virtual ~AutogradFunction() = default;
    virtual std::vector<Tensor> forward(const std::vector<Tensor>& inputs) = 0;
    virtual std::vector<Tensor> backward(const std::vector<Tensor>& grad_output) = 0;
    std::vector<Tensor> saved;
};

struct AutogradNode {
    std::shared_ptr<AutogradFunction> fn;
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    bool is_leaf = false;
};

class AutogradEngine {
public:
    static AutogradEngine& instance();
    void backward(Tensor& loss);
    void register_node(const std::shared_ptr<AutogradNode>& node);
    void clear();

private:
    AutogradEngine() = default;
    std::vector<std::shared_ptr<AutogradNode>> nodes_;
    std::unordered_map<void*, std::weak_ptr<AutogradNode>> output_to_node_;
};

Tensor matmul_grad(const Tensor& a, const Tensor& b, const Tensor& grad_output);
Tensor relu_grad(const Tensor& x, const Tensor& grad);
Tensor silu_grad(const Tensor& x, const Tensor& grad);
Tensor gelu_grad(const Tensor& x, const Tensor& grad);
Tensor softmax_grad(const Tensor& output, const Tensor& grad);
Tensor layer_norm_grad(const Tensor& x, const Tensor& gamma, const Tensor& grad, int N);
Tensor rms_norm_grad(const Tensor& x, const Tensor& gamma, const Tensor& grad, int N);

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets);
Tensor cross_entropy_grad(const Tensor& logits, const Tensor& targets);

} // namespace oil
