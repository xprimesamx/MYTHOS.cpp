#include "oil/model.h"
#include "oil/tokenizer.h"
#include "oil/generator.h"
#include "inference.h"

#include <iostream>
#include <string>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: oil_infer <model.oil> [prompt]" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    std::string prompt = argc > 2 ? argv[2] : "Hello, ";

    oil::DenseModel model;
    try {
        model.load(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    oil::BPETokenizer tokenizer;
    std::string vocab_path = model_path;
    size_t dot = vocab_path.rfind('.');
    if (dot != std::string::npos)
        vocab_path = vocab_path.substr(0, dot);
    vocab_path += ".vocab";

    try {
        if (std::filesystem::exists(vocab_path)) {
            tokenizer.load(vocab_path);
        } else {
            std::cerr << "Warning: vocab file not found at " << vocab_path
                      << ", using empty tokenizer" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading tokenizer: " << e.what() << std::endl;
        return 1;
    }

    oil::Generator gen(&model, &tokenizer);

    oil::SamplerConfig cfg;
    cfg.temperature = 0.7f;
    cfg.top_k = 40;
    cfg.top_p = 0.9f;
    cfg.max_tokens = 512;

    auto result = gen.generate_full(prompt, cfg);
    std::cout << result.text << std::endl;
    std::cerr << "Generated " << result.tokens_per_sec << " tok/s"
              << " (" << result.duration_sec << "s)"
              << std::endl;

    return 0;
}
