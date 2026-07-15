# `self_play.h` — Self-Play Training

**Path:** `include/oil/self_play.h`

Self-play reinforcement learning framework for improving model outputs through iterative play.

## SelfPlayConfig

```cpp
struct SelfPlayConfig {
    int num_episodes = 1000;
    int max_steps_per_episode = 128;
    float temperature = 0.8f;
    float exploration_rate = 0.1f;
    bool use_replay_buffer = true;
    int replay_buffer_size = 10000;
};
```

## SelfPlayTrainer

```cpp
class SelfPlayTrainer {
    SelfPlayConfig config;
    std::unique_ptr<Model> model;
    
    void train_episode();
    void self_play_round();
    Tensor generate_response(const Tensor& prompt);
    float compute_reward(const Tensor& response, const Tensor& target);
    void update_policy(const std::vector<Tensor>& trajectories);
    void add_to_replay(const Tensor& state, const Tensor& action, float reward);
    Tensor sample_from_replay(int batch_size);
};
```

### Training Loop

1. **Generate** — Model produces responses to prompts
2. **Evaluate** — Reward function scores the responses
3. **Learn** — Policy updated via reinforcement learning
4. **Repeat** — Model plays against itself iteratively

### Applications

- Output quality improvement through iterative self-play
- Alignment tuning without human labels
- Exploration of diverse generation strategies
- Reward model training
