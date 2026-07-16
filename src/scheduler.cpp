#include "oil/scheduler.h"
#include <algorithm>

namespace oil {

float LinearDecayScheduler::get_lr(int step) {
    float ratio = std::min(1.0f, (float)step / total_steps_);
    last_lr_ = start_lr_ + (end_lr_ - start_lr_) * ratio;
    return last_lr_;
}

float CosineDecayScheduler::get_lr(int step) {
    float ratio = std::min(1.0f, (float)step / total_steps_);
    last_lr_ = end_lr_ + (start_lr_ - end_lr_) * 0.5f *
               (1.0f + std::cos(3.14159265f * ratio));
    return last_lr_;
}

float ExponentialDecayScheduler::get_lr(int step) {
    int cycles = step / decay_steps_;
    last_lr_ = initial_lr_ * std::pow(decay_rate_, (float)cycles);
    return last_lr_;
}

float StepDecayScheduler::get_lr(int step) {
    int cycles = step / step_size_;
    last_lr_ = initial_lr_ * std::pow(gamma_, (float)cycles);
    return last_lr_;
}

float ReduceLROnPlateauScheduler::get_lr(int) {
    last_lr_ = base_lr_;
    return base_lr_;
}

void ReduceLROnPlateauScheduler::step_metric(float metric) {
    if (metric < best_metric_ - threshold_) {
        best_metric_ = metric;
        bad_epochs_ = 0;
    } else {
        bad_epochs_++;
        if (bad_epochs_ >= patience_) {
            base_lr_ = std::max(base_lr_ * factor_, min_lr_);
            bad_epochs_ = 0;
        }
    }
    steps_++;
}

float OneCycleScheduler::get_lr(int step) {
    float p = (float)step / total_steps_;
    if (p <= pct_start_) {
        float cycle_p = p / pct_start_;
        last_lr_ = init_lr_ + (max_lr_ - init_lr_) * cycle_p;
    } else {
        float cycle_p = (p - pct_start_) / (1.0f - pct_start_);
        last_lr_ = max_lr_ - (max_lr_ - final_lr_) * cycle_p;
    }
    return last_lr_;
}

float WarmupScheduler::get_lr(int step) {
    if (step < warmup_steps_) {
        float p = (float)step / warmup_steps_;
        last_lr_ = warmup_start_lr_ + (wrapped_->get_lr(0) - warmup_start_lr_) * p;
    } else {
        last_lr_ = wrapped_->get_lr(step - warmup_steps_);
    }
    return last_lr_;
}

float SequentialScheduler::get_lr(int step) {
    int cumulative = 0;
    for (auto& seg : segments_) {
        if (step < cumulative + seg.steps) {
            last_lr_ = seg.scheduler->get_lr(step - cumulative);
            return last_lr_;
        }
        cumulative += seg.steps;
    }
    if (!segments_.empty())
        last_lr_ = segments_.back().scheduler->get_lr(0);
    return last_lr_;
}

} // namespace oil