#include "oil/inference_opt.h"
#include "oil/model.h"
#include "oil/tensor.h"
#include "oil/math.h"
#include "oil/types.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>

using namespace oil;

static int g_tests = 0;
static int g_passed = 0;

#define CHECK(cond, msg) do { \
    g_tests++; \
    if (!(cond)) { printf("  FAIL: %s\n", msg); } \
    else { g_passed++; printf("  PASS: %s\n", msg); } \
} while(0)

#define CHECK_CLOSE(a, b, eps, msg) CHECK(std::fabs((a)-(b)) < (eps), msg)

static void test_paged_attention() {
    printf("\n=== D1: PagedAttention ===\n");
    PagedAttention pa(16, 4, 32);
    auto block = pa.alloc_block();
    CHECK(block.id >= 0, "alloc_block returns valid id");
    CHECK(block.k.numel() > 0, "alloc_block KV has data");
    CHECK(block.active, "alloc_block marked active");
    pa.free_block(block.id);

    Tensor Q({1, 4, 1, 16});
    Q.fill(1.0f);
    int64_t block_table[1] = {block.id};
    PagedAttention::Block blocks[1] = {block};
    blocks[0].k.fill(1.0f);
    blocks[0].v.fill(0.5f);
    auto out = pa.forward(Q, block_table, blocks, 1);
    CHECK(out.numel() > 0, "forward produces output");
    CHECK(std::isfinite(out.data<float>()[0]), "forward output is finite");

    Tensor Q_empty({1, 4, 0, 16});
    auto out2 = pa.forward(Q_empty, block_table, blocks, 0);
    CHECK(out2.numel() == 0, "empty query produces empty output");
}

static void test_speculative_decoder() {
    printf("\n=== D2: SpeculativeDecoder ===\n");
    SpeculativeDecoder dec(nullptr, nullptr, 4.0f);
    auto result = dec.generate({1, 2, 3}, 20);
    CHECK(result.size() >= 3, "generate preserves prompt");
    CHECK((int)result.size() <= 20, "generate respects max_tokens");
}

static void test_continuous_batching() {
    printf("\n=== D3: ContinuousBatching ===\n");
    ContinuousBatching cb(nullptr, 4);
    CHECK(!cb.has_pending(), "no pending initially");

    BatchRequest req{{1, 2, 3}, 10, 42};
    cb.add_request(req);
    CHECK(cb.has_pending(), "has pending after add");

    auto resp = cb.step();
    CHECK(resp.id == 42 || resp.id == req.id, "response has request id");
}

static void test_compressed_kv_cache() {
    printf("\n=== D4: CompressedKVCache ===\n");
    CompressedKVCache cache(16, 2, 64);
    CHECK(cache.size() == 0, "initial size is 0");

    Tensor k({64}), v({64});
    k.fill(1.0f);
    v.fill(2.0f);
    cache.append(0, k, v);
    CHECK(cache.size() == 1, "size after append is 1");

    auto k_out = cache.get_k(0, 0);
    auto v_out = cache.get_v(0, 0);
    CHECK(k_out.numel() == 64, "retrieved k shape");
    CHECK(v_out.numel() == 64, "retrieved v shape");
    for (int64_t i = 0; i < 64; i++) {
        CHECK(k_out.data<float>()[i] == 1.0f || k_out.data<float>()[i] == -1.0f,
              "compressed k values are binary");
    }
    cache.clear();
    CHECK(cache.size() == 0, "clear resets size");
}

static void test_prefix_cache() {
    printf("\n=== D5: PrefixCache ===\n");
    PrefixCache pc(4);
    int64_t match = pc.match_prefix({1, 2, 3, 4, 5});
    CHECK(match == -1, "no match in empty cache");

    pc.store({1, 2, 3}, 0);
    match = pc.match_prefix({1, 2, 3, 4});
    CHECK(match >= 0, "partial match found");

    auto* kv = pc.get_cache(0, 0);
    CHECK(kv != nullptr, "get_cache returns cache for stored entry");
    if (kv) {
        CHECK(kv->context_len() == 0, "empty cache after store");
    }
}

static void test_tree_decoder() {
    printf("\n=== D6: TreeDecoder ===\n");
    TreeDecoder td(nullptr, 2);
    auto tokens = td.decode({1, 2}, 10);
    CHECK(tokens.size() >= 2, "preserves prompt");
}

static void test_flash_decoding() {
    printf("\n=== D7: FlashDecoding ===\n");
    int64_t B=1, H=2, N=4, D=8;
    Tensor Q({B, H, N, D});
    Tensor K({B, H, N, D});
    Tensor V({B, H, N, D});
    Q.fill(1.0f); K.fill(0.5f); V.fill(0.3f);

    auto out = flash_decoding(Q, K, V);
    CHECK(out.dim(0) == B && out.dim(1) == H && out.dim(2) == N && out.dim(3) == D,
          "flash_decoding output shape matches input");
    for (int64_t i = 0; i < out.numel(); i++)
        CHECK(std::isfinite(out.data<float>()[i]), "flash_decoding output is finite");
}

static void test_int8_inference() {
    printf("\n=== D8: INT8Inference ===\n");
    INT8Inference i8(nullptr);
    Tensor inp({1, 4}), pos({1, 4});
    inp.fill(1.0f); pos.fill(0.0f);
    auto out = i8.forward(inp, pos);
    CHECK(out.numel() == inp.numel(), "INT8Inference forward returns tensor (null model)");
}

static void test_fp8_inference() {
    printf("\n=== D9: FP8Inference ===\n");
    FP8Inference fp8(nullptr);
    Tensor inp({1, 4}), pos({1, 4});
    inp.fill(1.0f); pos.fill(0.0f);
    auto out = fp8.forward(inp, pos);
    CHECK(out.numel() == inp.numel(), "FP8Inference forward returns tensor (null model)");
}

static void test_model_shard() {
    printf("\n=== D10: ModelShard ===\n");
    ModelShard shard(nullptr, 0, 4);
    Tensor inp({1, 4});
    inp.fill(1.0f);
    auto out = shard.forward(inp);
    CHECK(out.numel() == inp.numel(), "ModelShard forward returns tensor");
}

static void test_dynamic_batcher() {
    printf("\n=== D11: DynamicBatcher ===\n");
    DynamicBatcher db(nullptr);
    auto results = db.batch_generate({"hello", "world"}, 10);
    CHECK(results.size() == 2, "batch_generate returns all results");
    CHECK(!results[0].empty(), "first result non-empty");
    CHECK(!results[1].empty(), "second result non-empty");
}

static void test_request_scheduler() {
    printf("\n=== D12: RequestScheduler ===\n");
    RequestScheduler rs;
    CHECK(!rs.has_next(), "no requests initially");
    CHECK(rs.pending_count() == 0, "pending count 0 initially");

    Request r1{1, {1,2,3}, 5, 100};
    Request r2{2, {4,5,6}, 1, 200};
    rs.add(r1);
    rs.add(r2);
    CHECK(rs.has_next(), "has requests after add");
    CHECK(rs.pending_count() == 2, "pending count 2");

    auto next = rs.next();
    CHECK(next.id == 1 || next.id == 2, "next returns valid request");
}

static void test_inference_memory_pool() {
    printf("\n=== D13: InferenceMemoryPool ===\n");
    InferenceMemoryPool pool(64, 16);
    CHECK(pool.capacity() == 16, "pool capacity");
    CHECK(pool.used() == 0, "pool initially empty");

    void* p1 = pool.alloc();
    CHECK(p1 != nullptr, "pool alloc returns non-null");
    CHECK(pool.used() == 1, "pool usage after alloc");

    void* p2 = pool.alloc();
    CHECK(p2 != nullptr, "pool alloc second block");
    CHECK(p1 != p2, "consecutive allocs return different pointers");

    pool.free(p1);
    CHECK(pool.used() == 1, "pool usage after free");

    pool.free(p2);
    CHECK(pool.used() == 0, "pool usage after all freed");

    pool.free(nullptr);
    CHECK(pool.used() == 0, "free(nullptr) no-op");
}

static void test_stream_config() {
    printf("\n=== D14-D16: StreamConfig ===\n");
    StreamConfig cfg;
    CHECK(!cfg.stream, "default stream is false");
    CHECK(cfg.max_tokens == 512, "default max_tokens");
    CHECK(cfg.eos_id == -1, "default eos_id");

    DecodeResult dr;
    CHECK(dr.text.empty(), "default result text empty");
    CHECK(dr.perplexity == 0.0f, "default perplexity 0");

    cfg.stream = true;
    cfg.max_tokens = 128;
    CHECK(cfg.stream, "stream set to true");
}

static void test_logprobs() {
    printf("\n=== D17: Logprobs ===\n");
    Tensor logits({2, 4});
    logits.fill(0.0f);
    std::vector<int> tokens = {0, 1};
    auto probs = compute_logprobs(logits, tokens);
    CHECK(probs.size() == 2, "logprobs for each token");
    for (auto& p : probs) {
        CHECK(p.size() == 4, "logprobs for each vocab entry");
        float sum = 0;
        for (float v : p) sum += std::exp(v);
        CHECK_CLOSE(sum, 1.0f, 1e-4f, "exp(logprobs) sum to 1");
    }
}

static void test_embedding_endpoint() {
    printf("\n=== D18: EmbeddingEndpoint ===\n");
    EmbeddingEndpoint ep(nullptr);
    auto emb = ep.embed("hello world");
    CHECK(emb.numel() == 768, "embedding output is 768-dim (stub)");
}

static void test_reranker() {
    printf("\n=== D19: Reranker ===\n");
    Reranker rk(nullptr);
    float s = rk.score("query", "document");
    CHECK(s > 0, "reranker score positive");

    auto scores = rk.score_batch("query", {"doc1", "doc2", "doc3"});
    CHECK(scores.size() == 3, "batch reranker returns all scores");
    for (float s2 : scores) CHECK(s2 > 0, "individual scores positive");
}

static void test_grammar_decoder() {
    printf("\n=== D20: GrammarDecoder ===\n");
    GrammarDecoder gd("nonexistent.grammar");
    std::vector<float> logits(32000, 0.0f);
    logits[100] = 10.0f;
    auto constrained = gd.constrain(logits, {1, 2, 3});
    CHECK(!constrained.empty(), "constrain returns tokens");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("MYTHOS.cpp — D1-D20 Inference Optimization Test Suite\n");
    printf("=====================================================\n");

    test_paged_attention();
    test_speculative_decoder();
    test_continuous_batching();
    test_compressed_kv_cache();
    test_prefix_cache();
    test_tree_decoder();
    test_flash_decoding();
    test_int8_inference();
    test_fp8_inference();
    test_model_shard();
    test_dynamic_batcher();
    test_request_scheduler();
    test_inference_memory_pool();
    test_stream_config();
    test_logprobs();
    test_embedding_endpoint();
    test_reranker();
    test_grammar_decoder();

    printf("\n=====================================================\n");
    printf("Results: %d / %d tests passed", g_passed, g_tests);
    if (g_passed == g_tests) printf(" -- ALL PASSED\n");
    else printf(" (%d FAILED)\n", g_tests - g_passed);
    return (g_passed == g_tests) ? 0 : 1;
}
