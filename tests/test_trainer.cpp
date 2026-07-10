#include "oil/model.h"
#include "oil/transformer.h"
#include "oil/trainer.h"
#include "oil/optimizer.h"
#include "oil/autograd.h"
#include "oil/tokenizer.h"
#include "oil/tensor.h"
#include "oil/types.h"
#include "oil/math.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdio>

int main() {
    // Create a tiny model
    oil::TransformerConfig cfg;
    cfg.vocab_size = 50;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;
    cfg.ffn_hidden_size = 32;
    cfg.norm_eps = 1e-5f;
    cfg.max_seq_len = 32;

    oil::DenseModel model(cfg);
    oil::BPETokenizer tokenizer;

    // Test Trainer construction
    oil::Trainer trainer(&model, &tokenizer);
    oil::AdamW optimizer(1e-3f);
    trainer.compile(&optimizer);

    // Test forward+loss with random data
    {
        const int64_t B = 2;
        const int64_t S = 4;

        oil::Tensor input_ids(oil::Shape{B, S}, oil::DType::F32);
        oil::Tensor labels(oil::Shape{B, S}, oil::DType::F32);

        unsigned int seed = 42;
        for (int64_t i = 0; i < B * S; i++) {
            seed = seed * 1103515245u + 12345u;
            ((int*)input_ids.data())[i] = (int)((seed >> 16) % (cfg.vocab_size - 1)) + 1;
            ((int*)labels.data())[i] = (int)((seed >> 8) % cfg.vocab_size);
        }

        // Compute loss using cross_entropy_loss
        oil::Tensor logits = model.forward(input_ids, input_ids);
        float loss = oil::math::sum(oil::cross_entropy_loss(logits, labels));

        std::cout << "Initial loss: " << loss << std::endl;

        // Verify loss is finite
        assert(std::isfinite(loss));
        assert(!std::isnan(loss));

        // Verify loss is non-negative (cross-entropy is always >= 0)
        assert(loss >= 0.0f);
    }

    // Test train_step produces finite loss
    {
        const int64_t B = 1;
        const int64_t S = 4;

        oil::Tensor input_ids(oil::Shape{B, S}, oil::DType::F32);
        oil::Tensor labels(oil::Shape{B, S}, oil::DType::F32);

        for (int64_t i = 0; i < B * S; i++) {
            ((int*)input_ids.data())[i] = (int)((i * 3 + 1) % (cfg.vocab_size - 1)) + 1;
            ((int*)labels.data())[i] = (int)((i * 7) % cfg.vocab_size);
        }

        float loss = trainer.train_step(input_ids, labels);
        std::cout << "Train step loss: " << loss << std::endl;

        assert(std::isfinite(loss));
        assert(!std::isnan(loss));
        assert(loss >= 0.0f);
    }

    // Test DataLoader creation (with synthetic data)
    {
        // Create a temp data file
        std::string tmp_path = "_test_trainer_data.txt";
        {
            std::FILE* f = std::fopen(tmp_path.c_str(), "w");
            assert(f);
            for (int i = 0; i < 100; i++)
                std::fprintf(f, "hello world this is training data %d\n", i);
            std::fclose(f);
        }

        oil::BPETokenizer tokenizer2;
        std::vector<std::string> corpus = {"hello world this is training data"};
        tokenizer2.train(corpus, 32);

        oil::DataLoader dataloader(&tokenizer2, tmp_path, 2, 8);
        assert(dataloader.num_batches() > 0);

        oil::Tensor batch_ids(oil::Shape{2, 8}, oil::DType::F32);
        oil::Tensor batch_labels(oil::Shape{2, 8}, oil::DType::F32);

        bool has_data = dataloader.next_batch(batch_ids, batch_labels);
        assert(has_data);

        std::remove(tmp_path.c_str());

        // Test reset and shuffle
        dataloader.reset();
        assert(dataloader.num_batches() > 0);
        dataloader.shuffle();
    }

    // Test cross_entropy_grad produces finite values
    {
        const int64_t B = 2;
        const int64_t S = 4;
        const int64_t V = cfg.vocab_size;

        oil::Tensor logits(oil::Shape{B, S, V}, oil::DType::F32);
        oil::Tensor targets(oil::Shape{B, S}, oil::DType::F32);

        logits.fill(0.0f);
        for (int64_t i = 0; i < B * S; i++)
            ((int*)targets.data())[i] = (int)(i % V);

        auto grad = oil::cross_entropy_grad(logits, targets);
        assert(grad.shape() == logits.shape());
        for (int64_t i = 0; i < grad.numel(); i++) {
            assert(std::isfinite(grad.data<float>()[i]));
        }
    }

    // Test TrainConfig defaults
    {
        oil::TrainConfig cfg;
        assert(cfg.batch_size == 8);
        assert(cfg.seq_length == 512);
        assert(cfg.num_epochs == 3);
        assert(cfg.learning_rate == 3e-4f);
        assert(cfg.weight_decay == 1e-2f);
        assert(cfg.warmup_steps == 100);
    }

    std::cout << "All trainer tests passed!" << std::endl;
    return 0;
}
