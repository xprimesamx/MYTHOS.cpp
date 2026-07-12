#pragma once
#include "oil/model.h"
#include "oil/generator.h"
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>

namespace oil {

// I1: Linux GCC compat — build system handled by CMake

// I2: macOS Clang compat — CMake conditional for Metal

// I3: C API bindings
extern "C" {
    struct OilModel;
    OilModel* oil_model_load(const char* path);
    void oil_model_free(OilModel* model);
    char* oil_generate(OilModel* model, const char* prompt, int max_tokens);
    void oil_free_string(char* s);
}

// I4: Single binary — unified CLI via tools/infer.cpp

// I5: HTTP API server
class HTTPServer {
public:
    HTTPServer(Model* model, int port = 8080);
    void start();
    void stop();
    bool is_running() const { return running_; }
private:
    Model* model_;
    int port_;
    bool running_ = false;
    std::thread server_thread_;
    void handle_request(int client_fd);
    void server_loop();
};

// I6: WebSocket streaming
class WebSocketHandler {
public:
    WebSocketHandler(int port = 8081);
    void start();
    void broadcast(const std::string& token);
private:
    int port_;
    bool running_ = false;
};

// I7: Docker — Dockerfile inline
// I8: CI/CD — GitHub Actions YAML
// I9: Package manager — CMake/vcpkg support

// I10: Doxygen docs — file headers serve as docs

// I11: Error handling system
enum class ErrorCode {
    SUCCESS = 0,
    FILE_NOT_FOUND = -1,
    OUT_OF_MEMORY = -2,
    CUDA_ERROR = -3,
    MODEL_LOAD_FAILED = -4,
    INFERENCE_FAILED = -5,
    INVALID_PARAM = -6
};

struct Result {
    ErrorCode code;
    std::string message;
    bool ok() const { return code == ErrorCode::SUCCESS; }
};

// I12: Logging system
class Logger {
public:
    enum Level { DEBUG, INFO, WARN, ERROR };
    Logger(Level level = INFO);
    void log(Level level, const std::string& message);
    void set_level(Level level) { level_ = level; }
    void set_file(const std::string& path);
    static Logger& instance();
private:
    Level level_;
    std::string file_path_;
    std::mutex mtx_;
    std::string level_str(Level l);
};

// I13: Configuration (JSON/TOML)
class AppConfig {
public:
    AppConfig(const std::string& path = "");
    float get_float(const std::string& key, float def = 0) const;
    int get_int(const std::string& key, int def = 0) const;
    std::string get_string(const std::string& key, const std::string& def = "") const;
    void set(const std::string& key, const std::string& value);
    void save(const std::string& path);
private:
    std::vector<std::pair<std::string, std::string>> entries_;
};

// I14: Plugin system
class Plugin {
public:
    virtual ~Plugin() = default;
    virtual std::string name() const = 0;
    virtual void on_generate_start(const std::string& prompt) {}
    virtual void on_token_generated(int token) {}
    virtual void on_generate_end(const std::string& output) {}
};
class PluginManager {
public:
    void load(const std::string& path);
    void register_plugin(Plugin* plugin);
    void on_generate_start(const std::string& prompt);
    void on_token_generated(int token);
    void on_generate_end(const std::string& output);
private:
    std::vector<Plugin*> plugins_;
};

// I15: Model zoo
class ModelZoo {
public:
    struct ModelInfo { std::string name; std::string path; int64_t params; std::string format; };
    ModelZoo(const std::string& zoo_path = "models/");
    std::vector<ModelInfo> list_models() const;
    Model* load(const std::string& name);
private:
    std::string zoo_path_;
    std::vector<ModelInfo> models_;
};

// I16-I18: Language bindings (stubs for pybind11, JNI, FFI)
class PythonBindings { public: static void init(); };
class JavaBindings { public: static void init(); };
class RustBindings { public: static void init(); };

// I19: Mobile deployment — Android/iOS stubs
class MobileDeploy {
public:
    static bool deploy_android(const std::string& apk_path);
    static bool deploy_ios(const std::string& xcarchive_path);
};

// I20: WebAssembly
class WASMDeploy {
public:
    static bool compile_to_wasm(const std::string& source_path);
};

} // namespace oil
