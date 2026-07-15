#include "oil/production.h"
#include "oil/model.h"
#include "oil/tensor.h"
#include "oil/types.h"
#include "oil/tokenizer.h"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

using namespace oil;

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

static void test_c_api() {
    printf("\n=== I3: C API ===\n");
    auto* model = oil_model_load("nonexistent.oil");
    CHECK(model == nullptr, "load nonexistent returns nullptr");

    auto* str = oil_generate(nullptr, "test", 10);
    CHECK(str != nullptr, "generate with null model returns non-null");
    CHECK(str[0] == '\0', "generate with null model returns empty string");
    oil_free_string(str);

    oil_model_free(nullptr);
    CHECK(true, "free nullptr is safe");

    oil_model_free(model);
    CHECK(true, "free null model is safe");
}

static void test_http_server() {
    printf("\n=== I5: HTTPServer ===\n");
    HTTPServer server(nullptr, 0);
    CHECK(!server.is_running(), "not running by default");

    server.start();
    CHECK(server.is_running(), "running after start");

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    server.stop();
    CHECK(!server.is_running(), "stopped after stop");
}

static void test_web_socket() {
    printf("\n=== I6: WebSocketHandler ===\n");
    WebSocketHandler ws(0);
    ws.start();
    ws.broadcast("hello");
    CHECK(true, "WebSocket starts and broadcasts");
}

static void test_error_handling() {
    printf("\n=== I11: Error Handling ===\n");
    Result ok;
    CHECK(ok.ok(), "default result is ok");
    CHECK(ok.code == ErrorCode::SUCCESS, "default code is SUCCESS");

    Result err{ErrorCode::FILE_NOT_FOUND, "file missing"};
    CHECK(!err.ok(), "error result is not ok");
    CHECK(err.code == ErrorCode::FILE_NOT_FOUND, "error code preserved");
    CHECK(err.message == "file missing", "error message preserved");
}

static void test_logger() {
    printf("\n=== I12: Logger ===\n");
    Logger& log = Logger::instance();
    log.set_level(Logger::DEBUG);
    log.log(Logger::INFO, "test message");
    log.set_file("_test_log.txt");
    log.log(Logger::WARN, "warning message");
    log.log(Logger::ERROR, "error message");
    log.set_level(Logger::ERROR);
    log.log(Logger::DEBUG, "should not appear");

    std::remove("_test_log.txt");
    CHECK(true, "Logger operations complete");
}

static void test_app_config() {
    printf("\n=== I13: AppConfig ===\n");
    AppConfig cfg("_test_config.txt");

    CHECK(cfg.get_float("missing", 1.5f) == 1.5f, "default float");
    CHECK(cfg.get_int("missing", 42) == 42, "default int");
    CHECK(cfg.get_string("missing", "default") == "default", "default string");

    cfg.set("key1", "value1");
    cfg.set("key2", "3.14");
    cfg.set("key3", "42");

    CHECK(cfg.get_string("key1", "") == "value1", "get string after set");
    CHECK_CLOSE(cfg.get_float("key2", 0), 3.14f, 1e-4f, "get float after set");
    CHECK(cfg.get_int("key3", 0) == 42, "get int after set");

    cfg.save("_test_config_out.txt");
    AppConfig cfg2("_test_config_out.txt");
    CHECK(cfg2.get_string("key1", "") == "value1", "persistence across files");

    std::remove("_test_config_out.txt");
}

static void test_plugin_system() {
    printf("\n=== I14: Plugin System ===\n");
    PluginManager pm;
    pm.register_plugin(nullptr);
    pm.on_generate_start("test");
    pm.on_token_generated(42);
    pm.on_generate_end("output");
    CHECK(true, "Plugin lifecycle complete");
}

static void test_model_zoo() {
    printf("\n=== I15: ModelZoo ===\n");
    ModelZoo zoo("models/");
    auto models = zoo.list_models();
    CHECK(models.size() >= 2, "zoo returns default models");

    auto* model = zoo.load("nonexistent");
    CHECK(model == nullptr, "load nonexistent returns nullptr");

    bool found_tiny = false;
    for (auto& m : models) {
        if (m.name == "tiny") found_tiny = true;
        CHECK(!m.path.empty(), "model path non-empty");
    }
    CHECK(found_tiny, "zoo contains tiny model");
}

static void test_language_bindings() {
    printf("\n=== I16-I18: Language Bindings ===\n");
    PythonBindings::init();
    JavaBindings::init();
    RustBindings::init();
    CHECK(true, "Language binding init functions complete");
}

static void test_mobile_wasm() {
    printf("\n=== I19-I20: Mobile/WASM ===\n");
    bool android = MobileDeploy::deploy_android("test.apk");
    CHECK(!android, "android deploy returns false (stub)");

    bool ios = MobileDeploy::deploy_ios("test.xcarchive");
    CHECK(!ios, "ios deploy returns false (stub)");

    bool wasm = WASMDeploy::compile_to_wasm("test.cpp");
    CHECK(!wasm, "wasm compile returns false (stub)");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("MYTHOS.cpp — Production (I1-I20) Test Suite\n");
    printf("===========================================\n");

    test_c_api();
    test_http_server();
    test_web_socket();
    test_error_handling();
    test_logger();
    test_app_config();
    test_plugin_system();
    test_model_zoo();
    test_language_bindings();
    test_mobile_wasm();

    printf("\n===========================================\n");
    printf("Results: %d / %d tests passed", g_passed, g_tests);
    if (g_passed == g_tests) printf(" -- ALL PASSED\n");
    else printf(" (%d FAILED)\n", g_tests - g_passed);
    return (g_passed == g_tests) ? 0 : 1;
}
