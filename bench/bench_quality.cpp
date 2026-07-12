#include "oil/model.h"
#include "oil/tokenizer.h"
#include "oil/generator.h"
#include "oil/tensor.h"
#include "oil/types.h"
#include "oil/oil_format.h"

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>

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
        float fval = static_cast<float>(ids[i]);
        std::memcpy(input.data<float>() + i, &fval, sizeof(float));
        float tval = static_cast<float>(ids[i + 1]);
        std::memcpy(target.data<float>() + i, &tval, sizeof(float));
    }
    oil::Tensor positions(oil::Shape{1, N}, oil::DType::F32);
    for (int64_t i = 0; i < N; i++) {
        float pval = static_cast<float>(i);
        std::memcpy(positions.data<float>() + i, &pval, sizeof(float));
    }

    auto logits = model.forward(input, positions);
    float* ld = logits.data<float>();
    int64_t V = logits.shape().dims[2];

    double nll = 0.0;
    for (int64_t i = 0; i < N; i++) {
        float target_val;
        std::memcpy(&target_val, target.data<float>() + i, sizeof(float));
        int target_id = (int)target_val;
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

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: bench_quality <model.oil> <vocab.vocab> <eval.txt>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string vocab_path = argv[2];
    std::string eval_path = argv[3];

    std::ifstream efile(eval_path);
    if (!efile) { std::cerr << "Cannot open " << eval_path << std::endl; return 1; }
    std::stringstream buf;
    buf << efile.rdbuf();
    std::string eval_text = buf.str();

    oil::BPETokenizer tokenizer;
    tokenizer.load(vocab_path);

    oil::OILReader reader(model_path);
    auto config_data = reader.read_config();
    oil::TransformerConfig cfg;
    std::memcpy(&cfg, config_data.data(), std::min(config_data.size(), sizeof(cfg)));

    oil::DenseModel model(cfg);
    model.load(model_path);

    double ppl = compute_perplexity(model, tokenizer, eval_text);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nQuality Benchmarks:" << std::endl;
    std::cout << "  Model:      " << model_path << std::endl;
    std::cout << "  Perplexity: " << ppl << std::endl;
    return 0;
}
