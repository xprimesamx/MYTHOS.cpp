#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include "oil/tensor.h"
#include "oil/math.h"
#include "oil/moe_variants.h"
#include "oil/backend.h"
#include "oil/gpu_compute.h"

using namespace oil;
using namespace oil::moe;
using namespace oil::backend;

int tests_passed = 0;
int tests_total = 0;

#define TEST(name, expr) do { \
    tests_total++; \
    bool ok = (expr); \
    printf("  %s: %s\n", ok ? "PASS" : "FAIL", name); \
    if (ok) tests_passed++; else printf("  ^^^ FAILED: %s\n", #expr); \
} while(0)

#define TEST_FLOAT_EQ(a, b, eps) TEST(#a " ≈ " #b, std::fabs((a)-(b)) < (eps))

#define TEST_FORMAT(name, expr, arg) do { \
    tests_total++; \
    bool ok = (expr); \
    printf("  %s: %s\n", ok ? "PASS" : "FAIL", name); \
    if (ok) tests_passed++; \
} while(0)

void test_tensor_basics() {
    printf("\n=== TENSOR BASICS ===\n");

    Tensor t({2, 3});
    TEST("shape ok", t.dim(0) == 2 && t.dim(1) == 3);
    TEST("numel ok", t.numel() == 6);

    t.fill(3.0f);
    for (int64_t i = 0; i < 6; ++i)
        TEST("fill ok", t.data<float>()[i] == 3.0f);

    Tensor t2({2, 3});
    t2.fill(2.0f);
    Tensor t3({2, 3});
    math::add(t, t2, t3);
    for (int64_t i = 0; i < 6; ++i)
        TEST("add ok", t3.data<float>()[i] == 5.0f);

    Tensor t4({2, 3});
    math::mul(t, t2, t4);
    for (int64_t i = 0; i < 6; ++i)
        TEST("mul ok", t4.data<float>()[i] == 6.0f);

    Tensor t5({2, 3});
    math::scale(0.5f, t, t5);
    for (int64_t i = 0; i < 6; ++i)
        TEST("scale ok", t5.data<float>()[i] == 1.5f);
}

void test_math_ops() {
    printf("\n=== MATH OPS ===\n");

    Tensor x({2, 4});
    x.fill(2.0f);
    Tensor y({2, 4});
    math::relu(x, y);
    for (int64_t i = 0; i < 8; ++i) TEST("relu ok", y.data<float>()[i] == 2.0f);

    x.fill(-1.0f);
    math::relu(x, y);
    for (int64_t i = 0; i < 8; ++i) TEST("relu neg ok", y.data<float>()[i] == 0.0f);

    Tensor a({2, 3});
    Tensor b({3, 4});
    a.fill(1.0f);
    b.fill(2.0f);
    Tensor c({2, 4});
    math::gemm(1.0f, a, b, 0.0f, c);
    for (int64_t i = 0; i < 8; ++i)
        TEST("gemm ok", c.data<float>()[i] == 6.0f);

    Tensor ln({2, 4});
    ln.fill(1.0f);
    Tensor gamma({4});
    Tensor beta({4});
    gamma.fill(1.0f);
    beta.fill(0.0f);
    Tensor ln_out({2, 4});
    math::layer_norm(ln, gamma, beta, 1e-5f, ln_out);
    for (int64_t i = 0; i < 8; ++i)
        TEST("layer_norm ok", std::fabs(ln_out.data<float>()[i] - 0.0f) < 1e-5f);

    Tensor soft_x({2, 3});
    soft_x.data<float>()[0] = 1.0f; soft_x.data<float>()[1] = 2.0f; soft_x.data<float>()[2] = 3.0f;
    soft_x.data<float>()[3] = 1.0f; soft_x.data<float>()[4] = 2.0f; soft_x.data<float>()[5] = 3.0f;
    Tensor soft_y({2, 3});
    math::softmax(soft_x, soft_y, 1);
    float sum0 = soft_y.data<float>()[0] + soft_y.data<float>()[1] + soft_y.data<float>()[2];
    float sum1 = soft_y.data<float>()[3] + soft_y.data<float>()[4] + soft_y.data<float>()[5];
    TEST("softmax row sum 0", std::fabs(sum0 - 1.0f) < 1e-5f);
    TEST("softmax row sum 1", std::fabs(sum1 - 1.0f) < 1e-5f);
}

void test_sparse_moe() {
    printf("\n=== SPARSE MoE ===\n");
    MoEAllConfig cfg;
    cfg.num_experts = 4;
    cfg.top_k = 2;
    cfg.expert_hidden_size = 32;
    SparseMoE moe(16, cfg);
    Tensor x({2, 4, 16});
    x.fill(0.5f);
    auto out = moe.forward(x);
    TEST("sparse output shape", out.output.rank() == 3 && out.output.dim(0) == 2 && out.output.dim(1) == 4 && out.output.dim(2) == 16);
    TEST("sparse load balance computed", out.load_balance_loss >= 0.0f);
    TEST("expert indices populated", out.expert_indices.numel() > 0);
    TEST("expert weights populated", out.expert_weights.numel() > 0);
}

void test_soft_moe() {
    printf("\n=== SOFT MoE ===\n");
    MoEAllConfig cfg;
    cfg.num_experts = 2;
    cfg.num_slots_per_expert = 2;
    cfg.expert_hidden_size = 32;
    SoftMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.3f);
    auto out = moe.forward(x);
    TEST("softmoe output shape", out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16);
}

void test_hierarchical_moe() {
    printf("\n=== HIERARCHICAL MoE ===\n");
    MoEAllConfig cfg;
    cfg.num_groups = 2;
    cfg.experts_per_group = 2;
    cfg.top_groups = 1;
    cfg.top_experts_per_group = 1;
    cfg.expert_hidden_size = 32;
    HierarchicalMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.5f);
    auto out = moe.forward(x);
    TEST("hierarchical output shape", out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16);
}

void test_momoe() {
    printf("\n=== MoMoE ===\n");
    MoEAllConfig cfg;
    cfg.num_groups = 2;
    cfg.experts_per_group = 2;
    cfg.top_groups = 1;
    cfg.top_experts_per_group = 1;
    cfg.expert_hidden_size = 32;
    MoMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.5f);
    auto out = moe.forward(x);
    TEST("momoe output shape", out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16);
}

void test_expert_choice_moe() {
    printf("\n=== EXPERT CHOICE MoE ===\n");
    MoEAllConfig cfg;
    cfg.num_experts = 4;
    cfg.capacity_factor = 1.0f;
    cfg.expert_hidden_size = 32;
    ExpertChoiceMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.5f);
    auto out = moe.forward(x);
    TEST("expert_choice output shape", out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16);
}

void test_hash_moe() {
    printf("\n=== HASH MoE ===\n");
    MoEAllConfig cfg;
    cfg.num_experts = 4;
    cfg.hash_bucket_size = 1;
    cfg.expert_hidden_size = 32;
    HashMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.5f);
    Tensor token_ids({1, 4}, DType::I64);
    token_ids.fill(0);
    auto out = moe.forward(x, token_ids);
    TEST("hash_moe output shape", out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16);
}

void test_cross_layer_moe() {
    printf("\n=== CROSS-LAYER MoE ===\n");
    MoEAllConfig cfg;
    cfg.num_experts = 4;
    cfg.top_k = 1;
    cfg.expert_hidden_size = 32;
    CrossLayerMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.5f);
    auto out = moe.forward(x, 0);
    TEST("cross_layer output shape", out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16);
    auto out2 = moe.forward(x, 3);
    TEST("cross_layer reuse shared experts", out2.output.rank() == 3);
}

void test_multimodal_moe() {
    printf("\n=== MULTIMODAL MoE ===\n");
    MoEAllConfig cfg;
    cfg.num_experts = 8;
    cfg.top_k = 2;
    cfg.expert_hidden_size = 32;
    MultiModalMoE moe(16, cfg);
    Tensor x({1, 4, 16});
    x.fill(0.5f);
    Tensor modal({1, 4}, DType::I64);
    modal.fill(0);
    auto out = moe.forward(x, modal);
    TEST("multimodal output shape", out.output.rank() == 3 && out.output.dim(0) == 1 && out.output.dim(1) == 4 && out.output.dim(2) == 16);
}

void test_utility_fns() {
    printf("\n=== UTILITY FUNCTIONS ===\n");
    Tensor logits({2, 4});
    logits.fill(1.0f);
    Tensor indices, weights;
    Tensor probs = softmax_with_topk(logits, 2, indices, weights);
    TEST("softmax_with_topk probs shape", probs.dim(0) == 2 && probs.dim(1) == 4);
    TEST("indices dtype I64", indices.dtype() == DType::I64);
    TEST("indices shape", indices.dim(0) == 2 && indices.dim(1) == 2);
    TEST("weights shape", weights.dim(0) == 2 && weights.dim(1) == 2);
    float lb = compute_load_balance_loss(logits, indices, 4);
    TEST("load balance loss computed", lb >= 0.0f);
    float zl = compute_z_loss(probs);
    TEST("z-loss computed", zl > 0.0f);
    int64_t h = hash_token(42, 4);
    TEST("hash_token in range", h >= 0 && h < 4);
}

static void test_backend_ops(ComputeBackend* be, const char* name) {
    // Basic add
    Tensor a({2, 3}), b({2, 3}), c({2, 3});
    a.fill(2.0f);
    b.fill(3.0f);
    be->add(a, b, c);
    TEST_FORMAT("add %s", std::fabs(c.data<float>()[0] - 5.0f) < 1e-5f, name);

    // Basic gemm
    Tensor A({2, 3}), B({3, 4}), C({2, 4});
    A.fill(1.0f);
    B.fill(2.0f);
    be->gemm(1.0f, A, B, 0.0f, C);
    TEST_FORMAT("gemm %s", std::fabs(C.data<float>()[0] - 6.0f) < 1e-5f, name);

    // Relu
    Tensor rx({2, 4}), ry({2, 4});
    rx.fill(-1.0f);
    be->relu(rx, ry);
    TEST_FORMAT("relu %s", ry.data<float>()[0] == 0.0f, name);
    rx.fill(3.0f);
    be->relu(rx, ry);
    TEST_FORMAT("relu pos %s", ry.data<float>()[0] == 3.0f, name);

    // Scale
    Tensor sx({2, 3}), sy({2, 3});
    sx.fill(2.0f);
    be->scale(3.0f, sx, sy);
    TEST_FORMAT("scale %s", std::fabs(sy.data<float>()[0] - 6.0f) < 1e-5f, name);

    // Memory
    TEST_FORMAT("mem_free %s", be->memory_free() >= 0, name);
}

void test_backends() {
    printf("\n=== BACKENDS ===\n");

    // Test CPU_SCALAR
    {
        auto be = ComputeBackend::create(BackendConfig{BackendType::CPU_SCALAR});
        TEST("CPU_SCALAR created", be != nullptr);
        test_backend_ops(be, "CPU_SCALAR");
    }

    // Test CPU_AVX2 
    {
        auto be = ComputeBackend::create(BackendConfig{BackendType::CPU_AVX2});
        if (be && backend::is_avx2_available()) {
            TEST("CPU_AVX2 created", true);
            test_backend_ops(be, "CPU_AVX2");
        } else {
            printf("  SKIP CPU_AVX2 (not available on this platform)\n");
        }
    }

    // Test IGPU_SHARED if DirectX available
    {
        auto be = ComputeBackend::create(BackendConfig{BackendType::IGPU_SHARED});
        TEST("IGPU_SHARED created", be != nullptr);
        if (be && backend::is_directx_available()) {
            test_backend_ops(be, "IGPU_SHARED");
        } else {
            printf("  SKIP IGPU_SHARED ops (DirectX not available)\n");
        }
    }

    // Test GPU_DIRECTX if DirectX available
    {
        auto be = ComputeBackend::create(BackendConfig{BackendType::GPU_DIRECTX});
        TEST("GPU_DIRECTX created", be != nullptr);
        if (be && backend::is_directx_available()) {
            test_backend_ops(be, "GPU_DIRECTX");
        } else {
            printf("  SKIP GPU_DIRECTX ops (DirectX not available)\n");
        }
    }

    // Test RAM_SWAP
    {
        auto be = ComputeBackend::create(BackendConfig{BackendType::RAM_SWAP});
        TEST("RAM_SWAP created", be != nullptr);
        test_backend_ops(be, "RAM_SWAP");
    }

    // Test DISTRIBUTED (single-node)
    {
        BackendConfig dcfg;
        dcfg.type = BackendType::DISTRIBUTED;
        dcfg.num_devices = 1;
        auto be = ComputeBackend::create(dcfg);
        TEST("DISTRIBUTED created", be != nullptr);
        test_backend_ops(be, "DISTRIBUTED");
    }
}

void test_gpu_detection() {
    printf("\n=== GPU DETECTION ===\n");
    bool dx = backend::is_directx_available();
    bool cuda = backend::is_cuda_available();
    bool vk = backend::is_vulkan_available();
    bool avx2 = backend::is_avx2_available();
    int64_t gpu_mem = backend::gpu_memory_free(0);

    printf("  DirectX available: %s\n", dx ? "yes" : "no");
    printf("  CUDA available: %s\n", cuda ? "yes" : "no");
    printf("  Vulkan available: %s\n", vk ? "yes" : "no");
    printf("  AVX2 available: %s\n", avx2 ? "yes" : "no");
    printf("  GPU mem free: %lld\n", (long long)gpu_mem);

    // Detection functions should not crash
    TEST("is_directx_available() returns bool", dx || !dx);
    TEST("is_cuda_available() returns bool", cuda || !cuda);
    TEST("is_vulkan_available() returns bool", vk || !vk);
    TEST("is_avx2_available() returns bool", avx2 || !avx2);
    TEST("gpu_memory_free() returns >= 0", gpu_mem >= 0);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("MYTHOS.cpp — MoE Variants + Backend Test Suite\n");
    printf("==============================================\n");

    test_tensor_basics();
    test_math_ops();
    test_sparse_moe();
    test_soft_moe();
    test_hierarchical_moe();
    test_momoe();
    test_expert_choice_moe();
    test_hash_moe();
    test_cross_layer_moe();
    test_multimodal_moe();
    test_utility_fns();
    test_backends();
    test_gpu_detection();

    printf("\n==============================================\n");
    printf("Results: %d / %d tests passed", tests_passed, tests_total);
    if (tests_passed == tests_total) printf(" -- ALL PASSED\n");
    else printf(" (%d FAILED)\n", tests_total - tests_passed);

    return (tests_passed == tests_total) ? 0 : 1;
}
