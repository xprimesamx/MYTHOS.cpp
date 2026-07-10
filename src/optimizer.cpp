#include "oil/optimizer.h"
#include <cmath>
#include <algorithm>

namespace oil {

void Optimizer::add_param(Tensor* param) {
    parameters_.push_back(param);
}

void Optimizer::add_param_group(std::vector<Tensor*> params) {
    for (auto* p : params) add_param(p);
}

void Optimizer::zero_grad() {
    for (auto* p : parameters_) {
        if (p->requires_grad()) {
            p->set_grad(Tensor());
        }
    }
}

AdamW::AdamW(float lr, float beta1, float beta2, float eps, float weight_decay)
    : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay), t_(0) {}

void AdamW::set_lr(float lr) { lr_ = lr; }
float AdamW::get_lr() const { return current_lr_ > 0 ? current_lr_ : lr_; }
void AdamW::set_weight_decay(float wd) { weight_decay_ = wd; }
void AdamW::set_schedule(Schedule s, int warmup_steps, int total_steps) {
    schedule_ = s;
    warmup_steps_ = warmup_steps;
    total_steps_ = total_steps;
}

void AdamW::scheduler_step(int step) {
    float ratio = total_steps_ > 0 ? (float)step / total_steps_ : 1.0f;
    switch (schedule_) {
        case CONSTANT:
            current_lr_ = lr_;
            break;
        case COSINE:
            current_lr_ = lr_ * 0.5f * (1.0f + std::cos(3.14159265f * ratio));
            break;
        case LINEAR:
            current_lr_ = lr_ * (1.0f - ratio);
            break;
        case WARMUP_COSINE:
            if (step < warmup_steps_) {
                current_lr_ = lr_ * (float)step / warmup_steps_;
            } else {
                float cos_ratio = (float)(step - warmup_steps_) / (total_steps_ - warmup_steps_);
                current_lr_ = lr_ * 0.5f * (1.0f + std::cos(3.14159265f * cos_ratio));
            }
            break;
    }
    if (current_lr_ < 1e-8f) current_lr_ = 1e-8f;
}

void AdamW::zero_grad() { Optimizer::zero_grad(); }

void AdamW::step() {
    if (parameters_.empty()) return;
    t_++;
    scheduler_step(t_);
    float lr = current_lr_ > 0 ? current_lr_ : lr_;
    float bias_corr1 = 1.0f - std::pow(beta1_, t_);
    float bias_corr2 = 1.0f - std::pow(beta2_, t_);
    float lr_adj = lr * std::sqrt(bias_corr2) / bias_corr1;
    
    for (auto* param : parameters_) {
        if (!param->requires_grad() || param->grad().numel() == 0) continue;
        
        auto& state = state_[param];
        if (state.m.numel() == 0) {
            state.m = Tensor::zeros(param->shape());
            state.v = Tensor::zeros(param->shape());
        }
        
        float* m = (float*)state.m.data();
        float* v = (float*)state.v.data();
        const float* g = (const float*)param->grad().data();
        float* p = (float*)param->data();
        int64_t n = param->numel();
        
        // AdamW update
        for (int64_t i = 0; i < n; i++) {
            // Weight decay (decoupled)
            p[i] -= lr * weight_decay_ * p[i];
            
            // Biased moment estimates
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * g[i];
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * g[i] * g[i];
            
            // Bias-corrected update
            float m_hat = m[i] / bias_corr1;
            float v_hat = v[i] / bias_corr2;
            p[i] -= lr_adj * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

void SGD::zero_grad() { Optimizer::zero_grad(); }

SGD::SGD(float lr, float momentum, float weight_decay)
    : lr_(lr), momentum_(momentum), weight_decay_(weight_decay) {}

void SGD::step() {
    for (auto* param : parameters_) {
        if (!param->requires_grad() || param->grad().numel() == 0) continue;
        
        auto& state = state_[param];
        if (state.m.numel() == 0) {
            state.m = Tensor::zeros(param->shape());
        }
        
        float* m = (float*)state.m.data();
        const float* g = (const float*)param->grad().data();
        float* p = (float*)param->data();
        int64_t n = param->numel();
        
        for (int64_t i = 0; i < n; i++) {
            float g_reg = g[i] + weight_decay_ * p[i];
            m[i] = momentum_ * m[i] + lr_ * g_reg;
            p[i] -= m[i];
        }
    }
}

} // namespace oil
