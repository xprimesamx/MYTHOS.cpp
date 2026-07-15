#pragma once

#include "oil/tensor.h"
#include <unordered_map>
#include <vector>
#include <cmath>

namespace oil {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    void add_param(Tensor* param);
    void add_param_group(std::vector<Tensor*> params);

    struct ParamState {
        Tensor m, v;
    };

    ParamState& get_state(Tensor* p) { return state_[p]; }
    const std::unordered_map<Tensor*, ParamState>& all_state() const { return state_; }
    std::unordered_map<Tensor*, ParamState>& mutable_state() { return state_; }

protected:
    std::vector<Tensor*> parameters_;
    std::unordered_map<Tensor*, ParamState> state_;
};

class AdamW : public Optimizer {
public:
    AdamW(float lr = 3e-4f, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float weight_decay = 1e-2f);
    void step() override;
    void zero_grad() override;
    void set_lr(float lr);
    float get_lr() const;
    void set_weight_decay(float wd);

    void scheduler_step(int step);
    enum class Schedule { CONSTANT, COSINE, LINEAR, WARMUP_COSINE };
    void set_schedule(Schedule s, int warmup_steps = 0, int total_steps = 0);

private:
    float lr_, beta1_, beta2_, eps_, weight_decay_;
    int t_ = 0;
    Schedule schedule_ = Schedule::CONSTANT;
    int warmup_steps_ = 0;
    int total_steps_ = 0;
    float current_lr_ = 0;

    float get_lr_for_step(int step);
};

class SGD : public Optimizer {
public:
    SGD(float lr = 1e-3f, float momentum = 0.9f, float weight_decay = 0.0f);
    void step() override;
    void zero_grad() override;

private:
    float lr_, momentum_, weight_decay_;
};

} // namespace oil
