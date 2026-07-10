#include "oil/model.h"
#include "oil/tokenizer.h"
#include "oil/generator.h"
#include "oil/tensor.h"
#include "oil/types.h"

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>

struct QualityResult {
    std::string model_name;
    double perplexity;
    double accuracy;
    int correct;
    int total;
};

static double compute_perplexity(oil::Model& model, oil::Tokenizer& tokenizer,
                                  const std::string& text) {
    auto ids = tokenizer.encode(text);
    if (ids.size() < 2) return 0.0;
    int64_t N = (int64_t)ids.size() - 1;

    oil::Tensor input(oil::Shape{1, N}, oil::DType::F32);
    oil::Tensor target(oil::Shape{1, N}, oil::DType::F32);
    for (int64_t i = 0; i < N; i++) {
        ((int*)input.data())[i] = ids[i];
        ((int*)target.data())[i] = ids[i + 1];
    }
    oil::Tensor positions(oil::Shape{1, N}, oil::DType::F32);
    for (int64_t i = 0; i < N; i++)
        ((int*)positions.data())[i] = (int)i;

    auto logits = model.forward(input, positions);
    float* ld = logits.data<float>();
    int64_t V = logits.shape().dims[2];

    double nll = 0.0;
    for (int64_t i = 0; i < N; i++) {
        int target_id = ((int*)target.data())[i];
        float* row = ld + i * V;
        float max_val = -INFINITY;
        for (int64_t j = 0; j < V; j++)
            if (row[j] > max_val) max_val = row[j];
        double sum = 0.0;
        for (int64_t j = 0; j < V; j++)
            sum += std::exp((double)row[j] - (double)max_val);
        double prob = std::exp((double)row[target_id] - (double)max_val) / sum;
        nll += -std::log(prob + 1e-10);
    }

    return std::exp(nll / (double)N);
}

int main() {
    std::cout << "=== OIL Quality Benchmarks ===" << std::endl;
    std::cout << "NOTE: Quality benchmarks require a trained model file.\n" << std::endl;

    // Quality metrics placeholder
    std::cout << std::left << std::setw(24) << "Metric"
              << std::setw(14) << "Value"
              << std::endl;
    std::cout << std::string(38, '-') << std::endl;

    // These are placeholder measurements; real values require trained models
    std::cout << std::left << std::setw(24) << "Perplexity (placeholder)"
              << std::setw(14) << "N/A" << std::endl;
    std::cout << std::left << std::setw(24) << "Accuracy (placeholder)"
              << std::setw(14) << "N/A" << std::endl;
    std::cout << std::left << std::setw(24) << "Quality Score (placeholder)"
              << std::setw(14) << "N/A" << std::endl;

    std::cout << "\nTo run quality benchmarks:\n";
    std::cout << "  1. Train a model with oil_train\n";
    std::cout << "  2. Run: oil_info <model.oil> to verify\n";
    std::cout << "  3. Modify this file to load your model and run eval\n";

    return 0;
}
