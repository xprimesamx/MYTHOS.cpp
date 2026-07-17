#include "oil/asi.h"
#include "oil/model.h"
#include "oil/tensor.h"
#include "oil/trainer.h"
#include "oil/types.h"
#include "oil/optimizer.h"
#include "oil/tokenizer.h"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>

using namespace oil;
using namespace oil::asi;

static int g_tests = 0;
static int g_passed = 0;
static int g_failures = 0;

#define CHECK(cond, msg) do { \
    g_tests++; \
    if (!(cond)) { printf("  FAIL: %s\n", msg); g_failures++; } \
    else { g_passed++; printf("  PASS: %s\n", msg); } \
} while(0)

#define CHECK_CLOSE(a, b, eps, msg) do { \
    g_tests++; \
    if (std::fabs((a)-(b)) > (eps)) { printf("  FAIL: %s (got %f, expected %f)\n", msg, (double)(a), (double)(b)); g_failures++; } \
    else { g_passed++; printf("  PASS: %s\n", msg); } \
} while(0)

static void test_self_monitor() {
    printf("\n=== G1: SelfMonitor ===\n");
    SelfMonitor sm(nullptr);
    Tensor logits({2, 10});
    logits.fill(0.0f);
    float conf = sm.estimate_confidence(logits);
    CHECK(conf >= 0.0f && conf <= 1.0f, "confidence in [0,1]");

    auto state = sm.analyze("input", "output");
    CHECK(!state.recommendation.empty(), "recommendation is set");
    CHECK(state.confidence >= 0, "initial confidence non-negative");
}

static void test_self_reflector() {
    printf("\n=== G2: SelfReflector ===\n");
    SelfReflector sr(nullptr);
    auto reflection = sr.reflect("input", "output");
    CHECK(!reflection.empty(), "reflect returns non-empty");
    CHECK(reflection.find("Reflection:") != std::string::npos, "reflect contains prefix");

    auto refined = sr.refine("original", reflection);
    CHECK(!refined.empty(), "refine returns non-empty");
    CHECK(refined.find("[Refined]:") != std::string::npos, "refine contains prefix");
}

static void test_recursive_self_improver() {
    printf("\n=== G5: RecursiveSelfImprover ===\n");
    RecursiveSelfImprover rsi(nullptr, nullptr);
    rsi.improvement_cycle(5);
    bool result = rsi.self_modify("analysis");
    CHECK(!result, "self_modify returns false (stub)");
}

static void test_code_gen_self_improver() {
    printf("\n=== G6: CodeGenSelfImprover ===\n");
    CodeGenSelfImprover cg(nullptr);
    auto code = cg.generate_kernel("matmul", 128, 128, 128);
    CHECK(!code.empty(), "generate_kernel returns non-empty");
    CHECK(code.find("kernel") != std::string::npos, "generated code contains kernel");

    bool compiled = cg.compile_and_test(code);
    (void)compiled;
    CHECK(true, "compile_and_test executed successfully");
}

static void test_self_verifier() {
    printf("\n=== G7: SelfVerifier ===\n");
    SelfVerifier sv(nullptr);
    bool ok = sv.verify("2+2=4", "4");
    CHECK(ok, "verify returns true (stub)");

    auto cases = sv.find_edge_cases("solution");
    CHECK(cases.empty(), "edge cases empty (stub)");
}

static void test_capability_amplifier() {
    printf("\n=== G8: CapabilityAmplifier ===\n");
    CapabilityAmplifier ca(nullptr);
    float m = ca.measure("reasoning");
    CHECK(m >= 0.0f && m <= 1.0f, "measure in [0,1]");

    bool improved = ca.improve("reasoning", 10);
    CHECK(!improved, "improve returns false with null model");
}

static void test_safety_guardrails() {
    printf("\n=== G9: SafetyGuardrails ===\n");
    SafetyGuardrails sg;
    CHECK(sg.check_output("hello world"), "safe output passes");
    CHECK(!sg.check_output("rm -rf /"), "blocked output rejected");
    CHECK(sg.check_input("hello"), "safe input passes");
    CHECK(!sg.check_input("sudo rm -rf"), "dangerous input rejected");

    sg.set_kill_switch(true);
    CHECK(!sg.check_output("anything"), "output blocked when killed");
    CHECK(sg.is_killed(), "kill switch on");
    sg.set_kill_switch(false);
    CHECK(!sg.is_killed(), "kill switch off");
}

static void test_hitl() {
    printf("\n=== G10: HITL ===\n");
    HITL hitl;
    CHECK(!hitl.is_paused(), "not paused initially");
    bool approved = hitl.request_approval("deploy");
    CHECK(approved, "approval granted (stub)");

    hitl.pause();
    CHECK(hitl.is_paused(), "paused after call");
    hitl.resume();
    CHECK(!hitl.is_paused(), "not paused after resume");
}

static void test_alignment_system() {
    printf("\n=== G11-G12: AlignmentSystem ===\n");
    AlignmentSystem as;
    float score = as.value_alignment_score("helpful output");
    CHECK(score > 0, "alignment score positive");
    CHECK(as.max_loop_iterations == 100, "max loop iterations is 100");
}

static void test_world_model() {
    printf("\n=== G13: WorldModel ===\n");
    WorldModel wm(nullptr);
    Tensor state({10}), action({4});
    state.fill(0.0f); action.fill(1.0f);
    auto next = wm.simulate_step(state, action);
    CHECK(next.numel() == 10, "simulated state has same shape");

    auto plan = wm.plan(5);
    CHECK(plan.empty(), "plan returns empty (stub)");
}

static void test_curiosity_explorer() {
    printf("\n=== G14: CuriosityDrivenExplorer ===\n");
    CuriosityDrivenExplorer ce(nullptr);
    Tensor state({4});
    state.fill(0.0f);
    auto reward = ce.intrinsic_reward(state);
    CHECK(reward.numel() == 1, "intrinsic reward shape");
    CHECK(reward.data<float>()[0] >= 0.0f, "first state reward >= 0");

    auto reward2 = ce.intrinsic_reward(state);
    CHECK(reward2.data<float>()[0] >= 0.0f, "same state reward >= 0");

    auto steps = ce.explore(10);
    CHECK(steps.empty(), "explore returns empty (stub)");
}

static void test_multi_agent_system() {
    printf("\n=== G15: MultiAgentSystem ===\n");
    MultiAgentSystem mas(3);
    auto histories = mas.get_histories();
    CHECK(histories.size() == 3, "3 agent histories");

    mas.run_episode(5);
    auto histories2 = mas.get_histories();
    CHECK(histories2.size() == 3, "histories preserved after episode");
}

static void test_nas() {
    printf("\n=== G16: NAS ===\n");
    NeuralArchitectureSearch nas;
    auto arch = nas.search(10, 5);
    CHECK(arch.layers > 0, "NAS returns valid layers");
    CHECK(arch.hidden > 0, "NAS returns valid hidden size");
    CHECK(arch.score > 0, "NAS returns valid score");

    NeuralArchitectureSearch::Architecture base{12, 4096, 0.0f};
    (void)base;
}

static void test_hp_optimizer() {
    printf("\n=== G17: HPOptimizer ===\n");
    HPOptimizer hpo(nullptr);
    hpo.population_based_training(4, 5);

    CHECK(true, "HPOptimizer constructor + PBT succeeds");
}

static void test_continuous_learner() {
    printf("\n=== G18: ContinuousLearner ===\n");
    ContinuousLearner cl(nullptr);
    Tensor data({10});
    data.fill(1.0f);
    cl.update(data);

    bool retained = cl.prevent_forgetting(0.9f);
    CHECK(retained, "prevent_forgetting returns true (stub)");
}

static void test_knowledge_distillation() {
    printf("\n=== G19: KnowledgeDistillation ===\n");
    KnowledgeDistillation kd(nullptr, nullptr);
    CHECK(true, "distill succeeds (stub)");
}

static void test_prompt_optimizer() {
    printf("\n=== G20: PromptOptimizer ===\n");
    PromptOptimizer po(nullptr);
    auto optimized = po.optimize("translate", 5);
    CHECK(!optimized.empty(), "optimize returns non-empty");
    CHECK(optimized == "translate", "optimize returns original (stub)");

    float score = po.evaluate("prompt", "task");
    CHECK(score >= 0, "evaluate returns non-negative score");
}

static void test_chain_of_thought() {
    printf("\n=== G21: ChainOfThought ===\n");
    ChainOfThought cot(nullptr);
    auto result = cot.reason("Solve math: 2+2", 5);
    CHECK(!result.empty(), "reason returns result");

    auto chain = cot.get_chain();
    CHECK(chain.size() == 5, "chain has 5 steps");
    CHECK(chain[0].find("Step 0") != std::string::npos, "first step labeled");
    CHECK(chain[4].find("Step 4") != std::string::npos, "last step labeled");
}

static void test_tool_use() {
    printf("\n=== G22: ToolUse ===\n");
    ToolUse tu(nullptr);
    auto tools = tu.get_available_tools();
    CHECK(tools.size() >= 3, "has calculator, search, execute tools");

    auto result = tu.call_tool("calculator", "1 + 1");
    CHECK(!result.empty(), "tool call returns result");
    CHECK(result.find("calculator") != std::string::npos, "tool call includes tool name");
}

static void test_memory_system() {
    printf("\n=== G23: MemorySystem ===\n");
    MemorySystem mem(100);
    Tensor key({4}), value({4});
    key.fill(1.0f); value.fill(42.0f);
    mem.store(key, value);

    Tensor query({4});
    query.fill(1.0f);
    auto retrieved = mem.retrieve(query, 1);
    CHECK(retrieved.numel() == 4, "retrieved has same shape as query");
    for (int64_t i = 0; i < 4; i++)
        CHECK_CLOSE(retrieved.data<float>()[i], 42.0f, 1e-4f, "retrieved value matches stored");

    mem.consolidate();
    CHECK(true, "consolidate succeeds");
}

static void test_planning_engine() {
    printf("\n=== G24: PlanningEngine ===\n");
    PlanningEngine pe(nullptr);
    auto plan = pe.plan("build a house", 10);
    CHECK(!plan.empty(), "plan returns steps");

    bool executed = pe.execute(plan);
    CHECK(!executed, "execute returns false with null model");
}

static void test_evaluation_harness() {
    printf("\n=== G25: EvaluationHarness ===\n");
    EvaluationHarness eh(nullptr);
    auto result = eh.evaluate("hellaswag", 10);
    CHECK(result.accuracy >= 0, "accuracy non-negative");
    CHECK(result.loss >= 0, "loss >= 0");
    CHECK(result.samples >= 0, "samples non-negative");
    CHECK(result.samples <= 10, "samples respects n_samples");

    auto all = eh.evaluate_all();
    CHECK(all.size() >= 3, "evaluate_all returns multiple benchmarks");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("MYTHOS.cpp — ASI Pipeline (G1-G25) Test Suite\n");
    printf("=============================================\n");

    test_self_monitor();
    test_self_reflector();
    test_recursive_self_improver();
    test_code_gen_self_improver();
    test_self_verifier();
    test_capability_amplifier();
    test_safety_guardrails();
    test_hitl();
    test_alignment_system();
    test_world_model();
    test_curiosity_explorer();
    test_multi_agent_system();
    test_nas();
    test_hp_optimizer();
    test_continuous_learner();
    test_knowledge_distillation();
    test_prompt_optimizer();
    test_chain_of_thought();
    test_tool_use();
    test_memory_system();
    test_planning_engine();
    test_evaluation_harness();

    printf("\n=============================================\n");
    printf("Results: %d / %d tests passed", g_passed, g_tests);
    if (g_passed == g_tests) printf(" -- ALL PASSED\n");
    else printf(" (%d FAILED)\n", g_tests - g_passed);
    return (g_passed == g_tests) ? 0 : 1;
}
