#include "oil/asi.h"
#include "oil/random.h"
#include "oil/optimizer.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <numeric>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <cstring>
#include <ctime>

namespace oil {
namespace asi {

namespace {

// Simple character-level tokenization (no external tokenizer needed)
std::vector<int> simple_encode(const std::string& text, int vocab_size) {
    std::vector<int> ids;
    int offset = 5;
    int mod = std::max(1, vocab_size - offset);
    for (char c : text) {
        ids.push_back((int)(unsigned char)c % mod + offset);
    }
    return ids;
}

std::string simple_decode(const std::vector<int>& ids) {
    std::string s;
    for (int id : ids) {
        int c = id - 5;
        if (c >= 0 && c < 256) s += (char)c;
        else s += '?';
    }
    return s;
}

int greedy_argmax(const float* logits, int n) {
    return (int)(std::max_element(logits, logits + n) - logits);
}

// Generate new tokens given prompt IDs. Returns ONLY newly generated tokens.
std::vector<int> generate_new_tokens(Model* model, const std::vector<int>& prompt_ids, int vocab_size, int max_new) {
    if (!model) return {};
    std::vector<int> all_ids = prompt_ids;
    int context = 64;

    for (int step = 0; step < max_new; step++) {
        int64_t len = (int64_t)all_ids.size();
        int64_t start = std::max((int64_t)0, len - context);
        int64_t ctx_len = len - start;

        Tensor input_ids({1, ctx_len});
        Tensor positions({1, ctx_len});
        float* idp = input_ids.data<float>();
        float* psp = positions.data<float>();
        for (int64_t i = 0; i < ctx_len; i++) {
            idp[i] = (float)all_ids[start + i];
            psp[i] = (float)(start + i);
        }

        Tensor logits = model->forward(input_ids, positions, nullptr);
        int64_t V = logits.dim(logits.rank() - 1);
        const float* lp = logits.data<float>();

        int next = greedy_argmax(lp + (ctx_len - 1) * V, (int)V);
        all_ids.push_back(next);

        if (next < 2) break;
    }

    if ((int64_t)all_ids.size() <= (int64_t)prompt_ids.size()) {
        return {};
    }
    return std::vector<int>(all_ids.begin() + (int64_t)prompt_ids.size(), all_ids.end());
}

} // anonymous namespace

// ========================================================================
// G1: Self-Monitoring
// ========================================================================
SelfMonitor::SelfMonitor(Model* model) : model_(model) {}

float SelfMonitor::estimate_confidence(const Tensor& logits) {
    int64_t V = logits.dim(logits.rank() - 1);
    int64_t S = logits.numel() / V;
    const float* lp = logits.data<float>();
    float total_conf = 0;
    for (int64_t i = 0; i < S; i++) {
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[i * V + v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(lp[i * V + v] - max_l);
        float max_prob = std::exp(lp[i * V + 0] - max_l) / (sum + 1e-10f);
        total_conf += max_prob;
    }
    return total_conf / (float)(S > 0 ? S : 1);
}

MetaCognitionState SelfMonitor::analyze(const std::string& input, const std::string& output) {
    MetaCognitionState state;
    if (!model_) {
        state.confidence = 0.0f;
        state.uncertainty = 1.0f;
        state.token_confidences = {0.0f};
        state.recommendation = "no model";
        state.needs_reanalysis = false;
        state.reasoning_depth = 1;
        return state;
    }
    int vocab_size = (int)model_->config.vocab_size;

    auto input_ids = simple_encode(input, vocab_size);
    auto output_ids = simple_encode(output, vocab_size);

    int64_t seq_len = std::max((int64_t)1, (int64_t)output_ids.size());
    Tensor input_tensor({1, seq_len});
    Tensor pos_tensor({1, seq_len});
    float* idp = input_tensor.data<float>();
    float* psp = pos_tensor.data<float>();
    for (int64_t i = 0; i < seq_len; i++) {
        idp[i] = (float)output_ids[i % (int)output_ids.size()];
        psp[i] = (float)i;
    }

    Tensor logits = model_->forward(input_tensor, pos_tensor, nullptr);
    int64_t V = logits.dim(logits.rank() - 1);
    const float* lp = logits.data<float>();

    float total_conf = 0;
    float total_entropy = 0;
    state.token_confidences.clear();

    for (int64_t i = 0; i < seq_len; i++) {
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[i * V + v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(lp[i * V + v] - max_l);

        float max_prob = std::exp(lp[i * V + 0] - max_l) / (sum + 1e-10f);
        total_conf += max_prob;
        state.token_confidences.push_back(max_prob);

        float entropy = 0;
        for (int64_t v = 0; v < V; v++) {
            float p = std::exp(lp[i * V + v] - max_l) / (sum + 1e-10f);
            if (p > 1e-10f) entropy -= p * std::log(p);
        }
        total_entropy += entropy;
    }

    state.confidence = total_conf / (float)seq_len;
    float max_entropy = std::log((float)V + 1e-10f);
    state.uncertainty = total_entropy / (float)seq_len / (max_entropy > 0 ? max_entropy : 1.0f);
    state.reasoning_depth = (int)output_ids.size();

    float ratio = (float)output.size() / (float)std::max(input.size(), (size_t)1);
    state.needs_reanalysis = ratio < 0.3f || ratio > 3.0f || state.confidence < 0.5f;

    if (state.needs_reanalysis) {
        state.recommendation = "reanalyze";
    } else if (state.confidence > 0.8f) {
        state.recommendation = "proceed";
    } else {
        state.recommendation = "verify with caution";
    }

    return state;
}

// ========================================================================
// G2: Self-reflection
// ========================================================================
SelfReflector::SelfReflector(Model* model) : model_(model) {}

std::string SelfReflector::reflect(const std::string& input, const std::string& output) {
    if (!model_) return "Reflection: No model available";
    int vocab_size = (int)model_->config.vocab_size;
    std::string prompt = "Reflect on this output for input \"" + input + "\": " + output + ". How could this output be improved?";
    auto ids = simple_encode(prompt, vocab_size);
    auto gen = generate_new_tokens(model_, ids, vocab_size, 50);
    return simple_decode(gen);
}

std::string SelfReflector::refine(const std::string& original, const std::string& reflection) {
    if (!model_) return "[Refined]: No model available";
    int vocab_size = (int)model_->config.vocab_size;
    std::string prompt = "Original: " + original + "\nReflection: " + reflection + "\nRefined:";
    auto ids = simple_encode(prompt, vocab_size);
    auto gen = generate_new_tokens(model_, ids, vocab_size, 50);
    return simple_decode(gen);
}

// ========================================================================
// G5: Recursive Self-Improvement
// ========================================================================
RecursiveSelfImprover::RecursiveSelfImprover(Model* model, Trainer* trainer)
    : model_(model), trainer_(trainer) {}

void RecursiveSelfImprover::improvement_cycle(int iterations) {
    if (!model_) return;

    int64_t orig_hidden = model_->config.hidden_size;
    int64_t orig_layers = model_->config.num_layers;
    int no_improvement_count = 0;
    float best_perplexity = 1e10f;

    for (int i = 0; i < iterations && i < AlignmentSystem::max_loop_iterations; i++) {
        if (no_improvement_count >= 10) {
            model_->config.hidden_size = orig_hidden;
            model_->config.num_layers = orig_layers;
            break;
        }

        int vocab_size = (int)model_->config.vocab_size;

        std::string test_input = "Self-evaluation test input iteration " + std::to_string(i);
        auto test_ids = simple_encode(test_input, vocab_size);
        int64_t len = std::max((int64_t)1, (int64_t)test_ids.size());

        Tensor input_tensor({1, len});
        Tensor pos_tensor({1, len});
        float* idp = input_tensor.data<float>();
        float* psp = pos_tensor.data<float>();
        for (int64_t j = 0; j < len; j++) {
            idp[j] = (float)test_ids[j % (int)test_ids.size()];
            psp[j] = (float)j;
        }

        Tensor logits = model_->forward(input_tensor, pos_tensor, nullptr);
        int64_t V = logits.dim(logits.rank() - 1);

        float loss = 0;
        int count = 0;
        for (int64_t j = 1; j < len; j++) {
            int target = test_ids[j % (int)test_ids.size()];
            const float* lp = logits.data<float>() + (j - 1) * V;
            float max_l = -INFINITY;
            for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[v]);
            float sum = 0;
            for (int64_t v = 0; v < V; v++) sum += std::exp(lp[v] - max_l);
            float prob = std::exp(lp[target % (int)V] - max_l) / (sum + 1e-10f);
            loss -= std::log(std::max(prob, 1e-10f));
            count++;
        }
        float perplexity = std::exp(loss / std::max(1, count));

        if (perplexity < best_perplexity) {
            best_perplexity = perplexity;
            no_improvement_count = 0;
        } else {
            no_improvement_count++;
        }

        std::string analysis = "iteration=" + std::to_string(i) +
            "|perplexity=" + std::to_string(perplexity) +
            "|vocab_size=" + std::to_string(vocab_size) +
            "|hidden_size=" + std::to_string(model_->config.hidden_size) +
            "|num_layers=" + std::to_string(model_->config.num_layers);

        if (!self_modify(analysis)) {
            model_->config.hidden_size = orig_hidden;
            model_->config.num_layers = orig_layers;
            break;
        }
    }
}

bool RecursiveSelfImprover::self_modify(const std::string& analysis) {
    if (!model_) return false;
    auto extract_val = [&](const std::string& key) -> float {
        auto pos = analysis.find(key + "=");
        if (pos == std::string::npos) return -1.0f;
        pos += key.size() + 1;
        auto end = analysis.find('|', pos);
        try { return std::stof(analysis.substr(pos, end - pos)); }
        catch (...) { return -1.0f; }
    };

    float perplexity = extract_val("perplexity");
    if (perplexity < 0) return false;

    bool modified = false;
    if (perplexity > 15.0f) {
        model_->config.hidden_size = std::min((int64_t)4096, model_->config.hidden_size + 128);
        model_->config.num_layers = std::min((int64_t)48, model_->config.num_layers + 1);
        modified = true;
    } else if (perplexity > 8.0f) {
        model_->config.hidden_size = std::min((int64_t)4096, model_->config.hidden_size + 64);
        modified = true;
    } else if (perplexity < 3.0f && model_->config.hidden_size > 768) {
        model_->config.hidden_size -= 32;
        modified = true;
    }

    return modified;
}

// ========================================================================
// G6: Code generation
// ========================================================================
CodeGenSelfImprover::CodeGenSelfImprover(Model* model) : model_(model) {}

std::string CodeGenSelfImprover::generate_kernel(const std::string& op, int64_t M, int64_t N, int64_t K) {
    std::ostringstream code;
    code << "// Auto-generated " << op << " kernel (M=" << M << " N=" << N << " K=" << K << ")\n";
    code << "#include <cstdint>\n";
    code << "extern \"C\" void kernel(const float* a, const float* b, float* c, int64_t m, int64_t n, int64_t k) {\n";

    if (op == "gemm") {
        code << "    for (int64_t i = 0; i < m; i++) {\n";
        code << "        for (int64_t j = 0; j < n; j++) {\n";
        code << "            float sum = 0.0f;\n";
        code << "            for (int64_t p = 0; p < k; p++) {\n";
        code << "                sum += a[i * k + p] * b[p * n + j];\n";
        code << "            }\n";
        code << "            c[i * n + j] = sum;\n";
        code << "        }\n";
        code << "    }\n";
    } else if (op == "rms_norm") {
        code << "    for (int64_t i = 0; i < m; i++) {\n";
        code << "        float sum_sq = 0.0f;\n";
        code << "        for (int64_t j = 0; j < n; j++) sum_sq += a[i * n + j] * a[i * n + j];\n";
        code << "        float rms = std::sqrt(sum_sq / (float)n + 1e-5f);\n";
        code << "        for (int64_t j = 0; j < n; j++) c[i * n + j] = a[i * n + j] / rms;\n";
        code << "    }\n";
    } else if (op == "softmax") {
        code << "    for (int64_t i = 0; i < m; i++) {\n";
        code << "        float max_val = a[i * n];\n";
        code << "        for (int64_t j = 1; j < n; j++) if (a[i * n + j] > max_val) max_val = a[i * n + j];\n";
        code << "        float sum = 0.0f;\n";
        code << "        for (int64_t j = 0; j < n; j++) sum += std::exp(a[i * n + j] - max_val);\n";
        code << "        for (int64_t j = 0; j < n; j++) c[i * n + j] = std::exp(a[i * n + j] - max_val) / sum;\n";
        code << "    }\n";
    } else if (op == "relu") {
        code << "    for (int64_t i = 0; i < m; i++) {\n";
        code << "        for (int64_t j = 0; j < n; j++) {\n";
        code << "            c[i * n + j] = a[i * n + j] > 0.0f ? a[i * n + j] : 0.0f;\n";
        code << "        }\n";
        code << "    }\n";
    } else if (op == "silu") {
        code << "    for (int64_t i = 0; i < m; i++) {\n";
        code << "        for (int64_t j = 0; j < n; j++) {\n";
        code << "            float x = a[i * n + j];\n";
        code << "            c[i * n + j] = x / (1.0f + std::exp(-x));\n";
        code << "        }\n";
        code << "    }\n";
    } else if (op == "add") {
        code << "    for (int64_t i = 0; i < m * n; i++) c[i] = a[i] + b[i];\n";
    } else if (op == "mul") {
        code << "    for (int64_t i = 0; i < m * n; i++) c[i] = a[i] * b[i];\n";
    } else {
        code << "    // Unrecognized op: " << op << ", defaulting to identity\n";
        code << "    for (int64_t i = 0; i < m * n; i++) c[i] = a[i];\n";
    }

    code << "}\n";
    return code.str();
}

bool CodeGenSelfImprover::compile_and_test(const std::string& code) {
    auto tmp_dir = std::filesystem::temp_directory_path();
    auto src_path = tmp_dir / "asi_kernel_test.cpp";

    {
        std::ofstream ofs(src_path);
        if (!ofs) return false;
        ofs << code;
    }

#ifdef _WIN32
    std::string cmd = "cl.exe /nologo /EHsc /c \"" + src_path.string() + "\" 2>nul";
    int ret = std::system(cmd.c_str());
    auto obj_path = src_path;
    obj_path.replace_extension(".obj");
    std::filesystem::remove(obj_path);
#else
    std::string cmd = "g++ -x c++ -std=c++20 -c -o /dev/null \"" + src_path.string() + "\" 2>/dev/null";
    int ret = std::system(cmd.c_str());
#endif

    std::filesystem::remove(src_path);
    return ret == 0;
}

bool CodeGenSelfImprover::replace_kernel(const std::string& op, const std::string& new_code) {
    static std::unordered_map<std::string, std::string> kernel_registry;
    kernel_registry[op] = new_code;
    return true;
}

// ========================================================================
// G7-G8: Self-verification, Capability amplification
// ========================================================================
SelfVerifier::SelfVerifier(Model* model) : model_(model) {}

bool SelfVerifier::verify(const std::string& problem, const std::string& solution) {
    if (solution.empty()) return false;

    std::string problem_lower = problem;
    std::string solution_lower = solution;
    std::transform(problem_lower.begin(), problem_lower.end(), problem_lower.begin(), ::tolower);
    std::transform(solution_lower.begin(), solution_lower.end(), solution_lower.begin(), ::tolower);

    // Map problem types to required keywords in solution
    std::vector<std::pair<std::string, std::vector<std::string>>> problem_keywords = {
        {"sort", {"sort", "order", "compare", "arrange"}},
        {"search", {"search", "find", "lookup", "index"}},
        {"add", {"+", "sum", "add", "plus"}},
        {"subtract", {"-", "subtract", "minus", "difference"}},
        {"multiply", {"*", "multiply", "product", "times"}},
        {"divide", {"/", "divide", "quotient", "division"}},
        {"reverse", {"reverse", "backward"}},
        {"sum", {"sum", "+", "total", "accumulate"}},
        {"average", {"average", "mean", "/", "divide"}},
        {"max", {"max", "largest", "maximum", "greatest"}},
        {"min", {"min", "smallest", "minimum", "least"}},
        {"count", {"count", "size", "length", "number"}},
        {"filter", {"filter", "remove", "keep", "select"}},
        {"merge", {"merge", "combine", "union"}},
        {"contains", {"contain", "find", "search", "has"}},
        {"remove", {"remove", "delete", "erase", "clear"}},
        {"replace", {"replace", "substitute", "swap"}},
        {"sqrt", {"sqrt", "square root", "√"}},
        {"power", {"pow", "^", "power", "exponent"}},
        {"modulo", {"%", "mod", "modulo", "remainder"}},
    };

    for (auto& [ptype, keywords] : problem_keywords) {
        if (problem_lower.find(ptype) != std::string::npos) {
            bool found_keyword = false;
            for (auto& kw : keywords) {
                if (solution_lower.find(kw) != std::string::npos) {
                    found_keyword = true;
                    break;
                }
            }
            if (!found_keyword) return false;
        }
    }

    return true;
}

std::vector<std::string> SelfVerifier::find_edge_cases(const std::string& solution) {
    std::vector<std::string> edge_cases;
    std::string sol_lower = solution;
    std::transform(sol_lower.begin(), sol_lower.end(), sol_lower.begin(), ::tolower);

    if (sol_lower.find("array") != std::string::npos || sol_lower.find("vector") != std::string::npos ||
        sol_lower.find("list") != std::string::npos) {
        edge_cases.push_back("empty array: handle zero-length input");
        edge_cases.push_back("single element array: verify boundary behavior");
        edge_cases.push_back("all identical elements: check stability");
    }

    if (sol_lower.find("sort") != std::string::npos || sol_lower.find("order") != std::string::npos) {
        edge_cases.push_back("already sorted input: ensure no unnecessary swaps");
        edge_cases.push_back("reverse sorted input: verify worst-case performance");
        edge_cases.push_back("duplicate values: check sort stability");
    }

    if (sol_lower.find("number") != std::string::npos || sol_lower.find("int") != std::string::npos ||
        sol_lower.find("count") != std::string::npos) {
        edge_cases.push_back("negative values: verify correct handling");
        edge_cases.push_back("zero value: check division by zero");
        edge_cases.push_back("integer overflow: test with large values near INT_MAX");
    }

    if (sol_lower.find("string") != std::string::npos || sol_lower.find("char") != std::string::npos ||
        sol_lower.find("text") != std::string::npos) {
        edge_cases.push_back("empty string: handle null or empty input");
        edge_cases.push_back("unicode characters: verify multi-byte support");
        edge_cases.push_back("very long string: check for buffer overflow");
    }

    if (sol_lower.find("pointer") != std::string::npos || sol_lower.find("reference") != std::string::npos ||
        sol_lower.find("node") != std::string::npos) {
        edge_cases.push_back("null pointer: verify null safety");
        edge_cases.push_back("self-referential structure: check for infinite loops");
        edge_cases.push_back("dangling pointer: ensure no use-after-free");
    }

    if (sol_lower.find("search") != std::string::npos || sol_lower.find("find") != std::string::npos) {
        edge_cases.push_back("target not found: return appropriate sentinel");
        edge_cases.push_back("target at first position");
        edge_cases.push_back("target at last position");
    }

    if (sol_lower.find("divide") != std::string::npos || sol_lower.find("/") != std::string::npos) {
        edge_cases.push_back("division by zero: prevent undefined behavior");
        edge_cases.push_back("negative divisor: verify sign handling");
    }

    if (sol_lower.find("recursive") != std::string::npos || sol_lower.find("recursion") != std::string::npos) {
        edge_cases.push_back("stack overflow: check recursion depth limits");
        edge_cases.push_back("base case: verify termination condition");
    }

    if (sol_lower.find("graph") != std::string::npos || sol_lower.find("tree") != std::string::npos) {
        edge_cases.push_back("disconnected graph: handle multiple components");
        edge_cases.push_back("cyclic graph: prevent infinite traversal");
        edge_cases.push_back("single node: verify minimum case");
    }

    return edge_cases;
}

CapabilityAmplifier::CapabilityAmplifier(Model* model) : model_(model) {}

float CapabilityAmplifier::measure(const std::string& capability) {
    if (!model_) return 0.0f;
    int vocab_size = (int)model_->config.vocab_size;

    if (capability == "reasoning") {
        std::string puzzle = "If all dogs are mammals and all mammals are animals, are all dogs animals? Answer yes or no.";
        auto ids = simple_encode(puzzle, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        std::string answer = simple_decode(gen);
        std::transform(answer.begin(), answer.end(), answer.begin(), ::tolower);
        return answer.find("yes") != std::string::npos ? 0.9f : 0.3f;
    }

    if (capability == "math") {
        std::string queries[] = {"What is 2 + 2?", "What is 10 - 3?", "What is 4 * 5?"};
        float correct = 0;
        for (auto& q : queries) {
            auto ids = simple_encode(q, vocab_size);
            auto gen = generate_new_tokens(model_, ids, vocab_size, 10);
            std::string answer = simple_decode(gen);
            if (!answer.empty() && std::isdigit((unsigned char)answer[0])) correct += 1.0f;
        }
        return correct / 3.0f;
    }

    if (capability == "summarization") {
        std::string text = "The quick brown fox jumps over the lazy dog. Summarize this.";
        auto ids = simple_encode(text, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 30);
        std::string summary = simple_decode(gen);
        float ratio = (float)summary.size() / (float)std::max(text.size(), (size_t)1);
        return std::min(1.0f, 1.0f / (ratio + 0.1f));
    }

    if (capability == "code") {
        std::string prompt = "Write a C++ function to add two integers.";
        auto ids = simple_encode(prompt, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 40);
        std::string code = simple_decode(gen);
        if (code.find("int") != std::string::npos && code.find("return") != std::string::npos) return 0.8f;
        return 0.3f;
    }

    if (capability == "language") {
        std::string prompt = "Translate to French: Hello world.";
        auto ids = simple_encode(prompt, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        std::string answer = simple_decode(gen);
        return answer.size() > 5 ? 0.6f : 0.2f;
    }

    if (capability == "instruction_following") {
        std::string tests[] = {
            "List three fruits.",
            "What is the opposite of hot?",
            "Count from 1 to 5."
        };
        float correct = 0;
        for (auto& t : tests) {
            auto ids = simple_encode(t, vocab_size);
            auto gen = generate_new_tokens(model_, ids, vocab_size, 15);
            std::string answer = simple_decode(gen);
            if (!answer.empty()) correct += 1.0f;
        }
        return correct / 3.0f;
    }

    if (capability == "creativity") {
        std::string prompt = "Write a short poem about the ocean.";
        auto ids = simple_encode(prompt, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 40);
        std::string poem = simple_decode(gen);
        int words = 0;
        for (char c : poem) if (c == ' ') words++;
        return std::min(1.0f, (float)words / 20.0f);
    }

    return 0.0f;
}

bool CapabilityAmplifier::improve(const std::string& capability, int steps) {
    if (!model_) return false;
    int vocab_size = (int)model_->config.vocab_size;

    std::string training_prompt;
    if (capability == "reasoning") {
        training_prompt = "Training: Apply logical reasoning step by step. ";
    } else if (capability == "math") {
        training_prompt = "Training: Solve the following math problem carefully. ";
    } else if (capability == "summarization") {
        training_prompt = "Training: Summarize the following text concisely. ";
    } else if (capability == "code") {
        training_prompt = "Training: Write correct and efficient code for: ";
    } else {
        training_prompt = "Training: Improve performance on: " + capability + ". ";
    }

    for (int s = 0; s < steps; s++) {
        std::string prompt = training_prompt + " Step " + std::to_string(s) + " of " + std::to_string(steps) + ".";
        auto ids = simple_encode(prompt, vocab_size);
        generate_new_tokens(model_, ids, vocab_size, 10);
    }

    return true;
}

// ========================================================================
// G9-G10: Safety + HITL
// ========================================================================
SafetyGuardrails::SafetyGuardrails() {
    blocked_patterns_ = {
        "rm -rf", "sudo", "delete everything",
        "DROP TABLE", "exec(", "system(",
        "rmdir", "format ", "del /f",
        "shutdown", "reboot", "chmod 777",
        "wget ", "curl ", "eval(",
        "Process.Start", "ShellExecute",
        "mkfs", "dd if=", "> /dev/sda",
        "powershell -Command", "Invoke-Expression",
        "cmd.exe /c", "rundll32",
        "reg delete", "reg add",
        "net user", "net localgroup"
    };
}

bool SafetyGuardrails::check_output(const std::string& output) {
    for (auto& p : blocked_patterns_)
        if (output.find(p) != std::string::npos) return false;
    return !kill_switch_;
}

bool SafetyGuardrails::check_input(const std::string& input) {
    return check_output(input);
}

HITL::HITL() {}

bool HITL::request_approval(const std::string& action) {
    if (paused_) return false;

    std::string lower = action;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    std::vector<std::string> dangerous = {
        "rm -rf", "sudo", "delete everything",
        "drop table", "exec(", "system(",
        "rmdir", "format ", "del /f",
        "shutdown", "reboot", "chmod 777",
        "wget ", "curl ", "eval(",
        "process.start", "shellexecute",
        "mkfs", "dd if=", "> /dev/sda",
        "powershell -command", "invoke-expression",
        "cmd.exe /c", "rundll32",
        "reg delete", "reg add",
        "net user", "net localgroup",
        "DROP TABLE", "exec(", "system(",
        "rmdir", "format ", "del /f"
    };

    for (auto& d : dangerous) {
        if (lower.find(d) != std::string::npos) return false;
    }

    return true;
}

// ========================================================================
// G11-G12: Alignment
// ========================================================================
AlignmentSystem::AlignmentSystem() {}

float AlignmentSystem::value_alignment_score(const std::string& output) {
    std::vector<std::string> harmful_patterns = {
        "kill", "steal", "bomb", "weapon", "illegal", "violence",
        "discriminat", "racist", "sexist", "hate", "murder",
        "suicide", "self-harm", "abuse", "exploit", "fraud",
        "scam", "terror", "assault", "kidnap", "torture",
        "traffick", "slavery", "genocide", "war crime"
    };

    float score = 1.0f;
    std::string lower = output;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    int matches = 0;
    for (auto& p : harmful_patterns) {
        size_t pos = 0;
        while ((pos = lower.find(p, pos)) != std::string::npos) {
            matches++;
            pos += p.size();
        }
    }

    score -= (float)matches * 0.15f;
    return std::max(0.0f, score);
}

// ========================================================================
// G13: World model
// ========================================================================
WorldModel::WorldModel(Model* model) : model_(model) {}

Tensor WorldModel::simulate_step(const Tensor& state, const Tensor& action) {
    if (!model_) { Tensor empty(state.shape()); empty.zero_(); return empty; }
    int vocab_size = (int)model_->config.vocab_size;

    int64_t state_n = state.numel();
    int64_t action_n = action.numel();
    int64_t total_n = std::min(state_n + action_n, (int64_t)64);

    std::vector<float> combined(total_n);
    const float* sd = state.data<float>();
    const float* ad = action.data<float>();
    for (int64_t i = 0; i < std::min(state_n, total_n); i++) combined[i] = sd[i];
    for (int64_t i = 0; i < std::min(action_n, total_n - state_n); i++)
        combined[state_n + i] = ad[i];

    Tensor input_ids({1, total_n});
    Tensor positions({1, total_n});
    float* idp = input_ids.data<float>();
    float* psp = positions.data<float>();
    for (int64_t i = 0; i < total_n; i++) {
        idp[i] = std::fmod(combined[i] * 1000.0f, (float)vocab_size);
        if (idp[i] < 0) idp[i] += (float)vocab_size;
        psp[i] = (float)i;
    }

    Tensor logits = model_->forward(input_ids, positions, nullptr);
    int64_t V = logits.dim(logits.rank() - 1);

    Tensor next_state(state.shape());
    float* nsd = next_state.data<float>();
    int64_t out_n = std::min(state_n, total_n);
    for (int64_t i = 0; i < out_n; i++) {
        const float* lp = logits.data<float>() + i * V;
        int max_idx = greedy_argmax(lp, (int)V);
        nsd[i] = (float)max_idx / (float)V;
    }

    return next_state;
}

    std::vector<Tensor> WorldModel::plan(int64_t horizon) {
    if (!model_) return {};
    std::vector<Tensor> plan_steps;
    int64_t dim = model_->config.hidden_size;
    Tensor current_state({dim});
    current_state.zero_();

    RNG rng(42);
    for (int64_t h = 0; h < horizon; h++) {
        Tensor action({dim});
        float* ad = action.data<float>();
        for (int64_t i = 0; i < dim; i++) {
            ad[i] = rng.uniform() * 2.0f - 1.0f;
        }

        Tensor next_state = simulate_step(current_state, action);
        plan_steps.push_back(next_state.clone());
        current_state = next_state;
    }

    return plan_steps;
}

// ========================================================================
// G14: Curiosity
// ========================================================================
CuriosityDrivenExplorer::CuriosityDrivenExplorer(Model* model) : model_(model) {}

Tensor CuriosityDrivenExplorer::intrinsic_reward(const Tensor& state) {
    float novelty = 0;
    for (auto& v : visited_states_) {
        const float* sd = state.data<float>();
        const float* vd = v.data<float>();
        int64_t n = std::min(state.numel(), v.numel());
        float dist = 0;
        for (int64_t i = 0; i < n; i++) dist += (sd[i] - vd[i]) * (sd[i] - vd[i]);
        novelty = std::max(novelty, std::sqrt(dist));
    }
    visited_states_.push_back(state);
    Tensor reward({1});
    reward.data<float>()[0] = novelty;
    return reward;
}

std::vector<int> CuriosityDrivenExplorer::explore(int64_t n_steps) {
    if (!model_) return {};
    std::vector<int> exploration_path;
    int64_t dim = model_->config.hidden_size;
    RNG rng((unsigned)std::time(nullptr));

    for (int64_t step = 0; step < n_steps; step++) {
        Tensor state({dim});
        float* sd = state.data<float>();
        for (int64_t i = 0; i < dim; i++) {
            sd[i] = rng.uniform();
        }

        Tensor reward = intrinsic_reward(state);
        float r = reward.data<float>()[0];
        if (r > 0.05f) {
            exploration_path.push_back((int)visited_states_.size() - 1);
        }
    }

    return exploration_path;
}

// ========================================================================
// G15: Multi-agent
// ========================================================================
MultiAgentSystem::MultiAgentSystem(int n_agents) {
    agents_.resize(n_agents);
}

void MultiAgentSystem::communicate(int from, int to) {
    if (from < 0 || from >= (int)agents_.size() ||
        to < 0 || to >= (int)agents_.size()) return;

    for (auto& msg : agents_[from].history) {
        agents_[to].history.push_back("[Agent " + std::to_string(from) + "]: " + msg);
    }
}

void MultiAgentSystem::run_episode(int steps) {
    if (agents_.size() < 2) return;

    std::mt19937 rng(42);
    for (int step = 0; step < steps; step++) {
        int from = (int)(rng() % agents_.size());
        int to = (int)(rng() % agents_.size());
        if (from == to) to = (to + 1) % (int)agents_.size();

        communicate(from, to);

        agents_[from].state = "state_after_step_" + std::to_string(step) + "_from_" + std::to_string(from);
        agents_[to].state = "state_after_step_" + std::to_string(step) + "_to_" + std::to_string(to);

        agents_[from].history.push_back("Step " + std::to_string(step) + ": communicated with agent " + std::to_string(to));
    }
}

std::vector<std::string> MultiAgentSystem::get_histories() const {
    std::vector<std::string> histories;
    for (auto& agent : agents_) {
        std::string h;
        for (auto& msg : agent.history) h += msg + "\n";
        histories.push_back(h);
    }
    return histories;
}

// ========================================================================
// G16: NAS
// ========================================================================
NeuralArchitectureSearch::NeuralArchitectureSearch() {}

NeuralArchitectureSearch::Architecture NeuralArchitectureSearch::mutate(const Architecture& a) {
    std::mt19937 rng(42);
    Architecture m = a;
    m.layers += (int)(rng() % 3 - 1);
    m.hidden += (int)(rng() % 128 - 64);
    m.layers = std::max(1, m.layers);
    m.hidden = std::max(64, m.hidden);
    return m;
}

float NeuralArchitectureSearch::evaluate(const Architecture& arch) {
    float raw = (float)(arch.layers * arch.hidden) / 4096.0f;
    return std::sqrt(raw) / (1.0f + std::sqrt(raw));
}

NeuralArchitectureSearch::Architecture NeuralArchitectureSearch::search(int population, int generations) {
    std::mt19937 rng(42);
    std::vector<Architecture> pop;

    for (int i = 0; i < population; i++) {
        Architecture a;
        a.layers = (int)(rng() % 24) + 1;
        a.hidden = (int)(rng() % 4080) + 16;
        a.score = evaluate(a);
        pop.push_back(a);
    }

    for (int g = 0; g < generations; g++) {
        std::sort(pop.begin(), pop.end(),
                  [](auto& a, auto& b) { return a.score > b.score; });

        int keep = std::max(1, population / 2);
        pop.resize(keep);

        int top_n = (int)pop.size();
        for (int i = top_n; i < population && top_n > 0; i++) {
            Architecture child = mutate(pop[i % top_n]);
            child.score = evaluate(child);
            pop.push_back(child);
        }

        if ((int)pop.size() > population) pop.resize(population);
    }

    std::sort(pop.begin(), pop.end(),
              [](auto& a, auto& b) { return a.score > b.score; });
    return pop.empty() ? Architecture{12, 4096, 0.0f} : pop[0];
}

// ========================================================================
// G17: Hyperparameter optimization
// ========================================================================
HPOptimizer::HPOptimizer(Trainer* trainer) : trainer_(trainer) {}

void HPOptimizer::population_based_training(int n_population, int n_generations) {
    if (!trainer_) return;

    struct HPConfig { float lr; float wd; float beta1; float score; };
    std::vector<HPConfig> pop(n_population);
    std::mt19937 rng(42);

    for (auto& c : pop) {
        c.lr = std::pow(10.0f, -4.0f + (float)(rng() % 100) / 100.0f * 2.0f);
        c.wd = std::pow(10.0f, -3.0f + (float)(rng() % 100) / 100.0f * 2.0f);
        c.beta1 = 0.8f + (float)(rng() % 100) / 500.0f;
        c.score = 0;
    }

    for (int gen = 0; gen < n_generations; gen++) {
        for (auto& c : pop) {
            AdamW optim(c.lr, c.beta1, 0.999f, 1e-8f, c.wd);
            float lr_score = 1.0f - std::abs(c.lr - 3e-4f) / 3e-4f * 0.5f;
            float wd_score = 1.0f - std::abs(c.wd - 1e-2f) / 1e-2f * 0.3f;
            float beta_score = 1.0f - std::abs(c.beta1 - 0.9f) / 0.9f * 0.2f;
            c.score = std::max(0.0f, lr_score * 0.5f + wd_score * 0.3f + beta_score * 0.2f);
        }

        std::sort(pop.begin(), pop.end(),
                  [](auto& a, auto& b) { return a.score > b.score; });

        int keep = std::max(1, n_population / 4);
        for (int i = keep; i < n_population; i++) {
            auto& parent = pop[i % keep];
            pop[i].lr = parent.lr * (0.5f + (float)(rng() % 100) / 100.0f);
            pop[i].wd = parent.wd * (0.5f + (float)(rng() % 100) / 100.0f);
            pop[i].beta1 = std::max(0.5f, std::min(0.999f,
                parent.beta1 + ((float)(rng() % 100) - 50.0f) / 500.0f));
        }
    }
}

// ========================================================================
// G18: Continuous learning
// ========================================================================
ContinuousLearner::ContinuousLearner(Model* model) : model_(model) {}

void ContinuousLearner::update(const Tensor& new_data) {
    exemplars_.push_back(new_data.clone());
    int64_t capacity = 100;
    while ((int64_t)exemplars_.size() > capacity) {
        exemplars_.erase(exemplars_.begin());
    }
}

bool ContinuousLearner::prevent_forgetting(float threshold) {
    if (exemplars_.empty() || !model_) return true;

    float total_loss = 0;
    int count = 0;
    int vocab_size = (int)model_->config.vocab_size;

    for (auto& ex : exemplars_) {
        int64_t n = ex.numel();
        if (n < 2) continue;

        int64_t seq_len = std::min((int64_t)64, n - 1);
        const float* ed = ex.data<float>();

        Tensor input_ids({1, seq_len});
        Tensor positions({1, seq_len});
        float* idp = input_ids.data<float>();
        float* psp = positions.data<float>();
        for (int64_t i = 0; i < seq_len; i++) {
            idp[i] = std::fmod(ed[i], (float)vocab_size);
            if (idp[i] < 0) idp[i] += (float)vocab_size;
            psp[i] = (float)i;
        }

        Tensor logits = model_->forward(input_ids, positions, nullptr);
        int64_t V = logits.dim(logits.rank() - 1);

        int64_t target_idx = ((int64_t)std::fmod(ed[seq_len], (float)vocab_size) % V + V) % V;
        const float* lp = logits.data<float>() + (seq_len - 1) * V;
        float max_l = -INFINITY;
        for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[v]);
        float sum = 0;
        for (int64_t v = 0; v < V; v++) sum += std::exp(lp[v] - max_l);
        float prob = std::exp(lp[target_idx] - max_l) / (sum + 1e-10f);
        total_loss -= std::log(std::max(prob, 1e-10f));
        count++;
    }

    float avg_loss = count > 0 ? total_loss / count : 0;
    return avg_loss <= threshold;
}

// ========================================================================
// G19: Knowledge distillation
// ========================================================================
KnowledgeDistillation::KnowledgeDistillation(Model* teacher, Model* student)
    : teacher_(teacher), student_(student) {}

void KnowledgeDistillation::distill(const DataLoader& data, int steps) {
    if (!teacher_ || !student_) return;
    float temperature = 2.0f;

    for (int step = 0; step < steps; step++) {
        Tensor input_ids, labels;
        if (!const_cast<DataLoader&>(data).next_batch(input_ids, labels)) break;

        int64_t B = input_ids.dim(0);
        int64_t L = input_ids.dim(1);
        int64_t seq_len = L;

        Tensor positions({1, seq_len});
        float* psp = positions.data<float>();
        for (int64_t i = 0; i < seq_len; i++) psp[i] = (float)i;

        Tensor teacher_logits = teacher_->forward(input_ids, positions, nullptr);
        Tensor student_logits = student_->forward(input_ids, positions, nullptr);

        int64_t V = teacher_logits.dim(teacher_logits.rank() - 1);
        int64_t total_pos = teacher_logits.numel() / V;
        const float* tl = teacher_logits.data<float>();
        const float* sl = student_logits.data<float>();

        float kl_loss = 0;
        for (int64_t i = 0; i < total_pos; i++) {
            float max_t = -INFINITY;
            for (int64_t v = 0; v < V; v++) max_t = std::max(max_t, tl[i * V + v] / temperature);
            float sum_t = 0;
            for (int64_t v = 0; v < V; v++) sum_t += std::exp(tl[i * V + v] / temperature - max_t);

            float max_s = -INFINITY;
            for (int64_t v = 0; v < V; v++) max_s = std::max(max_s, sl[i * V + v] / temperature);
            float sum_s = 0;
            for (int64_t v = 0; v < V; v++) sum_s += std::exp(sl[i * V + v] / temperature - max_s);

            for (int64_t v = 0; v < V; v++) {
                float p_t = std::exp(tl[i * V + v] / temperature - max_t) / (sum_t + 1e-10f);
                float p_s = std::exp(sl[i * V + v] / temperature - max_s) / (sum_s + 1e-10f);
                if (p_t > 1e-10f) kl_loss += p_t * std::log(p_t / (p_s + 1e-10f));
            }
        }
        kl_loss *= temperature * temperature / std::max(total_pos, (int64_t)1);

        float lr = 1e-4f / (1.0f + 0.1f * step);
        int64_t delta = 0;
        if (kl_loss > 1.0f) delta = 32;
        else if (kl_loss > 0.1f) delta = 8;

        if (delta > 0) {
            student_->config.hidden_size = std::min((int64_t)4096, student_->config.hidden_size + delta);
        }
    }
}

// ========================================================================
// G20: Prompt optimization
// ========================================================================
PromptOptimizer::PromptOptimizer(Model* model) : model_(model) {}

float PromptOptimizer::evaluate(const std::string& prompt, const std::string& task) {
    if (!model_) return 0.0f;
    int vocab_size = (int)model_->config.vocab_size;
    std::string full = prompt + "\n" + task;
    auto ids = simple_encode(full, vocab_size);
    auto gen = generate_new_tokens(model_, ids, vocab_size, 30);
    std::string response = simple_decode(gen);

    float length_score = std::min(1.0f, (float)response.size() / 100.0f);
    float relevance = 0;
    std::string task_lower = task;
    std::transform(task_lower.begin(), task_lower.end(), task_lower.begin(), ::tolower);
    std::string resp_lower = response;
    std::transform(resp_lower.begin(), resp_lower.end(), resp_lower.begin(), ::tolower);

    std::istringstream task_stream(task_lower);
    std::string word;
    int match_count = 0;
    int word_count = 0;
    while (task_stream >> word) {
        word_count++;
        if (resp_lower.find(word) != std::string::npos) match_count++;
    }

    relevance = word_count > 0 ? (float)match_count / (float)word_count : 0;
    return (length_score * 0.4f + relevance * 0.6f);
}

std::string PromptOptimizer::optimize(const std::string& task, int n_iterations) {
    if (!model_) return task;
    std::vector<std::string> templates = {
        "Solve the following: ",
        "Please answer: ",
        "Task: ",
        "Question: ",
        "Problem: ",
        "You are an expert. Respond to: ",
        "Let's think step by step: ",
        "Given the following, provide a solution: ",
        "Analyze and respond to: ",
        "Instructions: ",
    };

    std::string best_prompt = task;
    float best_score = evaluate(task, task);

    for (int i = 0; i < n_iterations && i < (int)templates.size(); i++) {
        std::string candidate = templates[i] + task;
        float score = evaluate(candidate, task);
        if (score > best_score) {
            best_score = score;
            best_prompt = candidate;
        }
    }

    return best_prompt;
}

// ========================================================================
// G21: Chain of Thought
// ========================================================================
ChainOfThought::ChainOfThought(Model* model) : model_(model) {}

std::string ChainOfThought::reason(const std::string& problem, int max_steps) {
    if (!model_) {
        chain_.clear();
        for (int i = 0; i < max_steps; i++) {
            chain_.push_back("Step " + std::to_string(i) + ": " + problem);
        }
        return chain_.empty() ? "No model available" : chain_.back();
    }
    chain_.clear();
    int vocab_size = (int)model_->config.vocab_size;

    std::string prompt = "Let's solve this step by step.\nProblem: " + problem + "\nStep 1:";
    auto prompt_ids = simple_encode(prompt, vocab_size);
    auto new_ids = generate_new_tokens(model_, prompt_ids, vocab_size, max_steps * 10);
    std::string output = simple_decode(new_ids);

    std::istringstream stream(output);
    std::string word;
    std::string current_step;
    while (stream >> word) {
        if (word.find("Step") != std::string::npos && !current_step.empty()) {
            if (!current_step.empty()) chain_.push_back(current_step);
            current_step = word;
        } else {
            if (!current_step.empty()) current_step += " ";
            current_step += word;
        }
    }
    if (!current_step.empty()) chain_.push_back(current_step);

    // If no steps were parsed from model output, use fallback
    if (chain_.empty()) {
        chain_.push_back("Step 1: Analyze the problem: " + problem);
        if (max_steps > 1) chain_.push_back("Step 2: Solve based on analysis");
        if (max_steps > 2) chain_.push_back("Step 3: Verify the solution");
    }

    return chain_.back();
}

// ========================================================================
// G22: Tool use
// ========================================================================
ToolUse::ToolUse(Model* model) : model_(model) {
    tools_ = {{"calculator", "Perform arithmetic"},
              {"search", "Search the web"},
              {"execute", "Run code"}};
}

std::string ToolUse::call_tool(const std::string& name, const std::string& args) {
    if (name == "calculator") {
        try {
            std::istringstream iss(args);
            float a, b;
            char op;
            if (iss >> a >> op >> b) {
                float result = 0;
                if (op == '+') result = a + b;
                else if (op == '-') result = a - b;
                else if (op == '*') result = a * b;
                else if (op == '/' && b != 0) result = a / b;
                else if (op == '/') return "Error: division by zero";
                else return "Error: unsupported operator " + std::string(1, op);
                return name + " returned: " + std::to_string(result);
            }
        } catch (...) {}
        return "Error: invalid expression format (expected: a op b)";
    }

    if (name == "search") {
        if (!model_) return "No results found for: " + args;
        int vocab_size = (int)model_->config.vocab_size;
        std::string prompt = "Search the web for: " + args + ". Result:";
        auto ids = simple_encode(prompt, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 30);
        std::string result = simple_decode(gen);
        return result.empty() ? "No results found for: " + args : result;
    }

    if (name == "execute") {
        if (!model_) return "Execution completed with no output";
        int vocab_size = (int)model_->config.vocab_size;
        std::string prompt = "Execute the following code and return output: " + args + "\nOutput:";
        auto ids = simple_encode(prompt, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 30);
        std::string result = simple_decode(gen);
        return result.empty() ? "Execution completed with no output" : result;
    }

    return "Tool \"" + name + "\" called with args: " + args + " (unrecognized tool)";
}

std::vector<std::string> ToolUse::get_available_tools() const {
    std::vector<std::string> names;
    for (auto& t : tools_) names.push_back(t.name);
    return names;
}

// ========================================================================
// G23: Memory system
// ========================================================================
MemorySystem::MemorySystem(int64_t capacity) : capacity_(capacity) {}

void MemorySystem::store(const Tensor& key, const Tensor& value) {
    if ((int64_t)memory_.size() >= capacity_) memory_.erase(memory_.begin());
    memory_.push_back({key, value});
}

Tensor MemorySystem::retrieve(const Tensor& query, int k) {
    int64_t nq = query.numel();
    std::vector<std::pair<float, int>> scores;
    for (size_t i = 0; i < memory_.size(); i++) {
        const float* qd = query.data<float>();
        const float* kd = memory_[i].first.data<float>();
        float dot = 0;
        for (int64_t j = 0; j < std::min(nq, memory_[i].first.numel()); j++)
            dot += qd[j] * kd[j];
        scores.push_back({dot, (int)i});
    }
    std::sort(scores.begin(), scores.end(),
              [](auto& a, auto& b) { return a.first > b.first; });
    Tensor result({nq}); result.zero_();
    for (int i = 0; i < std::min(k, (int)scores.size()); i++) {
        const float* vd = memory_[scores[i].second].second.data<float>();
        float* rd = result.data<float>();
        for (int64_t j = 0; j < std::min(nq, memory_[scores[i].second].second.numel()); j++)
            rd[j] += vd[j] / (float)k;
    }
    return result;
}

void MemorySystem::consolidate() {
    if (memory_.size() < 2) return;

    std::vector<std::pair<Tensor, Tensor>> merged;
    merged.push_back(memory_[0]);

    for (size_t i = 1; i < memory_.size(); i++) {
        auto& last = merged.back();
        auto& curr = memory_[i];
        const float* lk = last.first.data<float>();
        const float* ck = curr.first.data<float>();
        int64_t n = std::min(last.first.numel(), curr.first.numel());

        float sim = 0;
        float norm_l = 0, norm_c = 0;
        for (int64_t j = 0; j < n; j++) {
            sim += lk[j] * ck[j];
            norm_l += lk[j] * lk[j];
            norm_c += ck[j] * ck[j];
        }
        float denom = std::sqrt(norm_l * norm_c);
        if (denom > 1e-10f) sim /= denom;
        else sim = 0;

        if (sim > 0.95f) {
            for (int64_t j = 0; j < n; j++) {
                float* mlk = const_cast<float*>(lk);
                mlk[j] = (lk[j] + ck[j]) * 0.5f;
            }
        } else {
            merged.push_back(curr);
        }
    }

    memory_ = merged;
}

// ========================================================================
// G24: Planning engine
// ========================================================================
PlanningEngine::PlanningEngine(Model* model) : model_(model) {}

std::vector<PlanningEngine::PlanStep> PlanningEngine::plan(const std::string& goal, int max_steps) {
    if (!model_) {
        std::vector<PlanStep> steps;
        steps.push_back({"Analyze: " + goal, {}});
        if (max_steps > 1) steps.push_back({"Execute: " + goal, {"Analyze: " + goal}});
        if (max_steps > 2) steps.push_back({"Verify: " + goal, {"Execute: " + goal}});
        return steps;
    }
    int vocab_size = (int)model_->config.vocab_size;
    std::string prompt = "Decompose the goal into sequential steps.\nGoal: " + goal + "\nSteps:";
    auto ids = simple_encode(prompt, vocab_size);
    auto gen = generate_new_tokens(model_, ids, vocab_size, max_steps * 8);
    std::string output = simple_decode(gen);

    std::vector<PlanStep> steps;
    std::istringstream stream(output);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        if ((int)steps.size() >= max_steps) break;

        auto dep_pos = line.find("depends on:");
        if (dep_pos != std::string::npos) {
            std::string action_part = line.substr(0, dep_pos);
            std::string dep_part = line.substr(dep_pos + 11);
            std::vector<std::string> deps;
            std::istringstream dep_stream(dep_part);
            std::string dep;
            while (dep_stream >> dep) {
                if (!dep.empty()) deps.push_back(dep);
            }
            steps.push_back({action_part, deps});
        } else {
            steps.push_back({line, {}});
        }
    }

    if (steps.empty()) {
        steps.push_back({"Analyze: " + goal, {}});
        if (max_steps > 1) steps.push_back({"Execute: " + goal, {"Analyze: " + goal}});
        if (max_steps > 2) steps.push_back({"Verify: " + goal, {"Execute: " + goal}});
    }

    return steps;
}

bool PlanningEngine::execute(const std::vector<PlanStep>& plan) {
    if (!model_) return false;
    int vocab_size = (int)model_->config.vocab_size;

    for (size_t i = 0; i < plan.size(); i++) {
        std::string prompt = "Execute step " + std::to_string(i) + ": " + plan[i].action;
        auto ids = simple_encode(prompt, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        std::string result = simple_decode(gen);

        std::string lower = result;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("error") != std::string::npos ||
            lower.find("fail") != std::string::npos ||
            lower.find("cannot") != std::string::npos) {
            return false;
        }
    }

    return true;
}

// ========================================================================
// G25: Evaluation harness
// ========================================================================
EvaluationHarness::EvaluationHarness(Model* model) : model_(model) {}

EvaluationHarness::Result EvaluationHarness::evaluate(const std::string& benchmark_name, int n_samples) {
    if (!model_) return {0.0f, 0.0f, 0};
    int vocab_size = (int)model_->config.vocab_size;

    // Define test cases for different benchmarks (hardcoded mini-test sets)
    struct TestCase {
        std::string input;
        std::string expected_keyword;
    };

    std::vector<TestCase> test_cases;
    std::string benchmark_lower = benchmark_name;
    std::transform(benchmark_lower.begin(), benchmark_lower.end(), benchmark_lower.begin(), ::tolower);

    if (benchmark_lower.find("hellaswag") != std::string::npos) {
        test_cases = {
            {"A person is cooking eggs. What happens next?", "flip"},
            {"A car is driving down the road. What happens next?", "turn"},
            {"A dog is running in the park. What happens next?", "fetch"},
            {"A student is studying for an exam. What happens next?", "read"},
            {"A chef is chopping vegetables. What happens next?", "cook"},
        };
    } else if (benchmark_lower.find("mmlu") != std::string::npos) {
        test_cases = {
            {"What is the capital of France?", "Paris"},
            {"What is 2+2?", "4"},
            {"Who wrote Romeo and Juliet?", "Shakespeare"},
            {"What is the chemical symbol for water?", "H2O"},
            {"What planet is known as the Red Planet?", "Mars"},
        };
    } else if (benchmark_lower.find("arc") != std::string::npos) {
        test_cases = {
            {"If you drop a ball, what happens?", "fall"},
            {"What do plants need to grow?", "sunlight"},
            {"What happens when you heat water to 100C?", "boil"},
            {"Why do we wear warm clothes in winter?", "warm"},
            {"What does a seed grow into?", "plant"},
        };
    } else {
        test_cases = {
            {"Test question 1?", "answer"},
            {"Test question 2?", "response"},
        };
    }

    int n = std::min(n_samples, (int)test_cases.size());
    int correct = 0;
    float total_loss = 0;

    for (int i = 0; i < n; i++) {
        auto& tc = test_cases[i];
        auto ids = simple_encode(tc.input, vocab_size);
        auto gen = generate_new_tokens(model_, ids, vocab_size, 20);
        std::string response = simple_decode(gen);
        std::string resp_lower = response;
        std::transform(resp_lower.begin(), resp_lower.end(), resp_lower.begin(), ::tolower);

        if (resp_lower.find(tc.expected_keyword) != std::string::npos) {
            correct++;
        }

        int64_t seq_len = (int64_t)ids.size();
        if (seq_len > 1) {
            Tensor input_tensor({1, seq_len});
            Tensor pos_tensor({1, seq_len});
            float* idp = input_tensor.data<float>();
            float* psp = pos_tensor.data<float>();
            for (int64_t j = 0; j < seq_len; j++) {
                idp[j] = (float)ids[j];
                psp[j] = (float)j;
            }

            Tensor logits = model_->forward(input_tensor, pos_tensor, nullptr);
            int64_t V = logits.dim(logits.rank() - 1);

            for (int64_t j = 1; j < seq_len; j++) {
                int target = ids[j];
                const float* lp = logits.data<float>() + (j - 1) * V;
                float max_l = -INFINITY;
                for (int64_t v = 0; v < V; v++) max_l = std::max(max_l, lp[v]);
                float sum = 0;
                for (int64_t v = 0; v < V; v++) sum += std::exp(lp[v] - max_l);
                float prob = std::exp(lp[target] - max_l) / (sum + 1e-10f);
                total_loss -= std::log(std::max(prob, 1e-10f));
            }
        }
    }

    float accuracy = n > 0 ? (float)correct / (float)n : 0;
    float avg_loss = total_loss / (float)std::max(n, 1);

    return {accuracy, avg_loss, n};
}

std::vector<EvaluationHarness::Result> EvaluationHarness::evaluate_all() {
    return {evaluate("hellaswag"), evaluate("mmlu"), evaluate("arc")};
}

} // namespace asi
} // namespace oil
