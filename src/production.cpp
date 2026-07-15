#include "oil/production.h"
#include "oil/transformer.h"
#include "oil/tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <thread>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <set>
#include <map>
#include <chrono>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#define SOCKET_ERRNO WSAGetLastError()
#define SOCKET_EWOULDBLOCK WSAEWOULDBLOCK
#define SOCKET_ECONNRESET WSAECONNRESET
#define SOCKET_ETIMEDOUT WSAETIMEDOUT
#else
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <dirent.h>
#include <sys/stat.h>
#include <signal.h>
#define SOCKET_ERRNO errno
#define SOCKET_EWOULDBLOCK EWOULDBLOCK
#define SOCKET_ECONNRESET ECONNRESET
#define SOCKET_ETIMEDOUT ETIMEDOUT
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define closesocket(fd) close(fd)
#endif

#ifdef ERROR
#undef ERROR
#endif

namespace oil {

// ========================================================================
// Internal helpers
// ========================================================================
namespace {

#ifdef _WIN32
    static bool g_wsa_initialized = false;

    bool socket_platform_init() {
        if (g_wsa_initialized) return true;
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
        g_wsa_initialized = true;
        return true;
    }

    void socket_platform_cleanup() {
        if (g_wsa_initialized) {
            WSACleanup();
            g_wsa_initialized = false;
        }
    }
#else
    bool socket_platform_init() {
        signal(SIGPIPE, SIG_IGN);
        return true;
    }
    void socket_platform_cleanup() {}
#endif

    bool set_socket_timeout_impl(int fd, int seconds) {
#ifdef _WIN32
        DWORD timeout = seconds * 1000;
        return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                          (const char*)&timeout, sizeof(timeout)) == 0;
#else
        struct timeval tv;
        tv.tv_sec = seconds;
        tv.tv_usec = 0;
        return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                          &tv, sizeof(tv)) == 0;
#endif
    }

    bool close_socket_impl(int fd) {
#ifdef _WIN32
        return closesocket(fd) == 0;
#else
        return close(fd) == 0;
#endif
    }

    // SHA-1 implementation (RFC 3174)
    struct SHA1_CTX {
        uint32_t state[5];
        uint64_t count;
        uint8_t buffer[64];

        static void init(SHA1_CTX* ctx) {
            ctx->state[0] = 0x67452301;
            ctx->state[1] = 0xEFCDAB89;
            ctx->state[2] = 0x98BADCFE;
            ctx->state[3] = 0x10325476;
            ctx->state[4] = 0xC3D2E1F0;
            ctx->count = 0;
        }

        static uint32_t rotl32(uint32_t x, int n) {
            return (x << n) | (x >> (32 - n));
        }

        static void process_block(SHA1_CTX* ctx, const uint8_t block[64]) {
            uint32_t w[80];
            for (int i = 0; i < 16; i++) {
                w[i] = ((uint32_t)block[i*4] << 24) |
                       ((uint32_t)block[i*4+1] << 16) |
                       ((uint32_t)block[i*4+2] << 8) |
                       ((uint32_t)block[i*4+3]);
            }
            for (int i = 16; i < 80; i++) {
                w[i] = rotl32(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
            }

            uint32_t a = ctx->state[0];
            uint32_t b = ctx->state[1];
            uint32_t c = ctx->state[2];
            uint32_t d = ctx->state[3];
            uint32_t e = ctx->state[4];

            auto round = [&](int t) {
                uint32_t f, k;
                if (t < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
                else if (t < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
                else if (t < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
                else { f = b ^ c ^ d; k = 0xCA62C1D6; }

                uint32_t temp = rotl32(a, 5) + f + e + k + w[t];
                e = d; d = c; c = rotl32(b, 30); b = a; a = temp;
            };

            for (int t = 0; t < 80; t++) round(t);

            ctx->state[0] += a;
            ctx->state[1] += b;
            ctx->state[2] += c;
            ctx->state[3] += d;
            ctx->state[4] += e;
        }

        static void update(SHA1_CTX* ctx, const uint8_t* data, size_t len) {
            size_t idx = ctx->count & 63;
            ctx->count += len;
            size_t part = 64 - idx;
            if (len >= part) {
                memcpy(ctx->buffer + idx, data, part);
                process_block(ctx, ctx->buffer);
                for (size_t i = part; i + 63 < len; i += 64)
                    process_block(ctx, data + i);
                idx = 0;
            }
            memcpy(ctx->buffer + idx, data + (len - (len - idx) % 64),
                   (len - idx) % 64);
        }

        static void final(SHA1_CTX* ctx, uint8_t out[20]) {
            uint64_t bits = ctx->count * 8;
            uint8_t pad = 0x80;
            update(ctx, &pad, 1);
            while ((ctx->count & 63) != 56) {
                uint8_t zero = 0;
                update(ctx, &zero, 1);
            }
            for (int i = 7; i >= 0; i--) {
                uint8_t byte = (uint8_t)(bits >> (i * 8));
                update(ctx, &byte, 1);
            }
            for (int i = 0; i < 5; i++) {
                out[i*4]   = (uint8_t)(ctx->state[i] >> 24);
                out[i*4+1] = (uint8_t)(ctx->state[i] >> 16);
                out[i*4+2] = (uint8_t)(ctx->state[i] >> 8);
                out[i*4+3] = (uint8_t)(ctx->state[i]);
            }
        }
    };

    static thread_local std::string g_last_error;

} // anonymous namespace

// ========================================================================
// I3: C API
// ========================================================================
struct OilModel { Model* model; };

const char* oil_last_error() {
    return g_last_error.c_str();
}

bool file_exists(const char* path) {
#ifdef _WIN32
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
#endif
}

OilModel* oil_model_load(const char* path) {
    g_last_error.clear();
    if (!path) {
        g_last_error = "path is null";
        errno = EINVAL;
        return nullptr;
    }

    if (!file_exists(path)) {
        g_last_error = std::string("file not found: ") + path;
#ifdef _WIN32
        SetLastError(ERROR_FILE_NOT_FOUND);
#endif
        errno = ENOENT;
        return nullptr;
    }

    auto* om = new(std::nothrow) OilModel;
    if (!om) {
        g_last_error = "out of memory";
        errno = ENOMEM;
        return nullptr;
    }

    om->model = new(std::nothrow) DenseModel;
    if (!om->model) {
        delete om;
        g_last_error = "out of memory";
        errno = ENOMEM;
        return nullptr;
    }

    try {
        om->model->load(path);
    } catch (const std::exception& e) {
        g_last_error = std::string("model load failed: ") + e.what();
        delete om->model;
        delete om;
#ifdef _WIN32
        SetLastError(ERROR_FILE_NOT_FOUND);
#endif
        errno = ENOENT;
        return nullptr;
    } catch (...) {
        g_last_error = "model load failed: unknown error";
        delete om->model;
        delete om;
        errno = ENOENT;
        return nullptr;
    }

    return om;
}

void oil_model_free(OilModel* model) {
    if (!model) return;
    delete model->model;
    delete model;
}

char* oil_generate(OilModel* model, const char* prompt, int max_tokens) {
    g_last_error.clear();

    if (!model) {
        g_last_error = "model is null";
        errno = EINVAL;
        char* empty = new char[1];
        empty[0] = '\0';
        return empty;
    }
    if (!model->model) {
        g_last_error = "internal model is null";
        errno = EINVAL;
        char* empty = new char[1];
        empty[0] = '\0';
        return empty;
    }
    if (!prompt) {
        g_last_error = "prompt is null";
        errno = EINVAL;
        char* empty = new char[1];
        empty[0] = '\0';
        return empty;
    }
    if (max_tokens <= 0) {
        max_tokens = 64;
    }
    if (max_tokens > 4096) {
        max_tokens = 4096;
    }

    BPETokenizer bpe;
    std::vector<int> tokens;
    try {
        tokens = bpe.encode(prompt);
    } catch (const std::exception& e) {
        g_last_error = std::string("tokenization failed: ") + e.what();
        errno = EINVAL;
        char* empty = new char[1];
        empty[0] = '\0';
        return empty;
    }

    if (tokens.empty()) tokens = {1};

    std::vector<int> output;
    output.reserve(max_tokens);

    for (int i = 0; i < max_tokens; i++) {
        try {
            int64_t seq_len = (int64_t)(tokens.size() + output.size());
            Tensor input({1, seq_len});
            float* id = input.data<float>();
            for (size_t j = 0; j < tokens.size(); j++) id[j] = (float)tokens[j];
            for (size_t j = 0; j < output.size(); j++)
                id[tokens.size() + j] = (float)output[j];
            Tensor pos({1, seq_len});
            Tensor logits = model->model->forward(input, pos);
            if (logits.numel() == 0) break;

            int64_t V = logits.dim(logits.rank() - 1);
            const float* ld = logits.data<float>() + logits.numel() - V;
            int next = 0;
            for (int64_t v = 1; v < V; v++)
                if (ld[v] > ld[next]) next = (int)v;

            if (next == 2) break;
            output.push_back(next);
        } catch (const std::exception& e) {
            g_last_error = std::string("inference failed: ") + e.what();
            errno = EIO;
            break;
        }
    }

    std::string result = bpe.decode(output);
    char* cstr = new char[result.size() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

void oil_free_string(char* s) {
    delete[] s;
}

// ========================================================================
// I5: HTTP Server
// ========================================================================
HTTPServer::HTTPServer(Model* model, int port)
    : model_(model), port_(port) {
    thread_pool_size_ = 4;
    timeout_seconds_ = 30;
    max_body_size_ = 4 * 1024 * 1024;
}

HTTPServer::~HTTPServer() { stop(); }

void HTTPServer::set_timeout_seconds(int sec) {
    timeout_seconds_ = (std::max)(1, sec);
}

void HTTPServer::set_max_body_size(size_t bytes) {
    max_body_size_ = bytes;
}

bool HTTPServer::platform_init() {
    return socket_platform_init();
}

void HTTPServer::platform_cleanup() {
    socket_platform_cleanup();
}

bool HTTPServer::set_socket_timeout(int fd, int seconds) {
    return set_socket_timeout_impl(fd, seconds);
}

void HTTPServer::close_socket(int fd) {
    close_socket_impl(fd);
}

void HTTPServer::start() {
    if (running_) return;
    if (!platform_init()) return;
    running_ = true;
    stop_requested_ = false;
    server_thread_ = std::thread(&HTTPServer::server_loop, this);
}

void HTTPServer::stop() {
    running_ = false;
    stop_requested_ = true;
    queue_cv_.notify_all();
    if (server_thread_.joinable()) server_thread_.join();
    for (auto& t : worker_threads_) {
        if (t.joinable()) t.join();
    }
    worker_threads_.clear();
    platform_cleanup();
}

void HTTPServer::server_loop() {
#ifdef _WIN32
    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) { running_ = false; return; }
#else
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { running_ = false; return; }
#endif

    int opt = 1;
#ifdef _WIN32
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR,
               (const char*)&opt, sizeof(opt));
#else
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((u_short)port_);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        closesocket(server_fd);
        running_ = false;
        return;
    }

    if (listen(server_fd, 64) < 0) {
        closesocket(server_fd);
        running_ = false;
        return;
    }

    // Spawn worker threads
    int n_workers = (std::max)(1, thread_pool_size_);
    for (int i = 0; i < n_workers; i++) {
        worker_threads_.emplace_back(&HTTPServer::worker_loop, this);
    }

    fd_set read_fds;
    while (running_ && !stop_requested_) {
        FD_ZERO(&read_fds);
#ifdef _WIN32
        FD_SET(server_fd, &read_fds);
#else
        FD_SET(server_fd, &read_fds);
#endif
        struct timeval tv = {1, 0};

        int sel = select((int)server_fd + 1, &read_fds, nullptr, nullptr, &tv);
        if (sel < 0) {
            if (SOCKET_ERRNO == SOCKET_EWOULDBLOCK) continue;
            break;
        }
        if (sel == 0) continue;

        sockaddr_in client_addr;
#ifdef _WIN32
        int addr_len = sizeof(client_addr);
#else
        socklen_t addr_len = sizeof(client_addr);
#endif
        int client_fd = (int)accept(server_fd, (sockaddr*)&client_addr, &addr_len);
        if (client_fd < 0) continue;

        // Enqueue for worker thread
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            conn_queue_.push({client_fd});
        }
        queue_cv_.notify_one();
    }

    closesocket(server_fd);

    // Wait for workers to finish
    stop_requested_ = true;
    queue_cv_.notify_all();
}

void HTTPServer::worker_loop() {
    while (!stop_requested_) {
        ClientConnection conn;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::seconds(1), [this] {
                return !conn_queue_.empty() || stop_requested_;
            });
            if (stop_requested_ || conn_queue_.empty()) continue;
            conn = conn_queue_.front();
            conn_queue_.pop();
        }

        // Set receive timeout
        set_socket_timeout(conn.fd, timeout_seconds_);

        // Handle the request
        handle_request(conn.fd);

        // Close connection
        close_socket(conn.fd);
    }
}

std::string HTTPServer::get_mime_type(const std::string& path) const {
    std::string ext;
    auto dot = path.find_last_of('.');
    if (dot != std::string::npos) {
        ext = path.substr(dot);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
    }
    if (ext == ".html" || ext == ".htm") return "text/html";
    if (ext == ".json") return "application/json";
    if (ext == ".js")   return "application/javascript";
    if (ext == ".css")  return "text/css";
    if (ext == ".png")  return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".gif")  return "image/gif";
    if (ext == ".svg")  return "image/svg+xml";
    if (ext == ".txt")  return "text/plain";
    if (ext == ".xml")  return "application/xml";
    if (ext == ".wasm") return "application/wasm";
    return "text/plain";
}

void HTTPServer::send_response(int client_fd, int status,
                                const std::string& content_type,
                                const std::string& body) {
    std::string status_text;
    switch (status) {
        case 200: status_text = "OK"; break;
        case 201: status_text = "Created"; break;
        case 204: status_text = "No Content"; break;
        case 301: status_text = "Moved Permanently"; break;
        case 400: status_text = "Bad Request"; break;
        case 401: status_text = "Unauthorized"; break;
        case 403: status_text = "Forbidden"; break;
        case 404: status_text = "Not Found"; break;
        case 405: status_text = "Method Not Allowed"; break;
        case 413: status_text = "Payload Too Large"; break;
        case 429: status_text = "Too Many Requests"; break;
        case 500: status_text = "Internal Server Error"; break;
        case 502: status_text = "Bad Gateway"; break;
        case 503: status_text = "Service Unavailable"; break;
        default:  status_text = "Unknown"; break;
    }

    std::string response =
        "HTTP/1.1 " + std::to_string(status) + " " + status_text + "\r\n"
        "Content-Type: " + content_type + "\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n"
        "Server: MYTHOS.cpp\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "\r\n" + body;

    send(client_fd, response.c_str(), (int)response.size(), 0);
}

void HTTPServer::send_stream_response(int client_fd, const std::string& prompt,
                                       int max_tokens) {
    // SSE-style streaming response
    std::string headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Server: MYTHOS.cpp\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";

    send(client_fd, headers.c_str(), (int)headers.size(), 0);

    if (!model_) {
        std::string err = "data: {\"error\":\"no model loaded\"}\n\n";
        send(client_fd, err.c_str(), (int)err.size(), 0);
        return;
    }

    BPETokenizer bpe;
    std::vector<int> tokens;
    try {
        tokens = bpe.encode(prompt);
    } catch (...) {
        std::string err = "data: {\"error\":\"tokenization failed\"}\n\n";
        send(client_fd, err.c_str(), (int)err.size(), 0);
        return;
    }
    if (tokens.empty()) tokens = {1};

    std::vector<int> output;
    output.reserve(max_tokens);

    for (int i = 0; i < max_tokens; i++) {
        try {
            int64_t seq_len = (int64_t)(tokens.size() + output.size());
            Tensor input({1, seq_len});
            float* id = input.data<float>();
            for (size_t j = 0; j < tokens.size(); j++) id[j] = (float)tokens[j];
            for (size_t j = 0; j < output.size(); j++)
                id[tokens.size() + j] = (float)output[j];
            Tensor pos({1, seq_len});
            Tensor logits = model_->forward(input, pos);
            if (logits.numel() == 0) break;

            int64_t V = logits.dim(logits.rank() - 1);
            const float* ld = logits.data<float>() + logits.numel() - V;
            int next = 0;
            for (int64_t v = 1; v < V; v++)
                if (ld[v] > ld[next]) next = (int)v;

            if (next == 2) break;
            output.push_back(next);

            // Send token as SSE event
            std::string token_text = bpe.decode({next});
            std::string sse = "data: {\"token\":\"" + token_text + "\"}\n\n";
            if (send(client_fd, sse.c_str(), (int)sse.size(), 0) < 0) {
                break;
            }
        } catch (...) {
            break;
        }
    }

    // Send completion event
    std::string done = "data: {\"done\":true}\n\n";
    send(client_fd, done.c_str(), (int)done.size(), 0);
}

void HTTPServer::handle_request(int client_fd) {
    std::vector<char> full_buf;
    full_buf.reserve(8192);
    char tmp[4096];

    // Read request with timeout
    int n;
    bool timeout = false;
    auto start_time = std::chrono::steady_clock::now();

    while (full_buf.size() < max_body_size_) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
            >= timeout_seconds_) {
            timeout = true;
            break;
        }

#ifdef _WIN32
        n = recv(client_fd, tmp, sizeof(tmp) - 1, 0);
#else
        n = recv(client_fd, tmp, sizeof(tmp) - 1, 0);
#endif
        if (n < 0) {
            int err = SOCKET_ERRNO;
            if (err == SOCKET_EWOULDBLOCK || err == SOCKET_ETIMEDOUT) {
                timeout = true;
            }
            break;
        }
        if (n == 0) break;
        tmp[n] = '\0';
        full_buf.insert(full_buf.end(), tmp, tmp + n);

        // Check if we have the full headers
        std::string current(full_buf.data(), full_buf.size());
        auto header_end = current.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            // Check Content-Length
            auto cl_pos = current.find("Content-Length:");
            if (cl_pos == std::string::npos)
                cl_pos = current.find("content-length:");
            if (cl_pos != std::string::npos) {
                auto num_start = current.find_first_of("0123456789", cl_pos + 15);
                if (num_start != std::string::npos) {
                    int content_length = std::stoi(current.substr(num_start));
                    size_t body_received = full_buf.size() - header_end - 4;
                    if (body_received >= (size_t)content_length) break;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    if (full_buf.empty()) {
        send_response(client_fd, 400, "text/plain", "Empty request");
        return;
    }
    if (timeout) {
        send_response(client_fd, 408, "text/plain",
                       "{\"error\":\"request timeout\"}");
        return;
    }

    std::string request(full_buf.data(), full_buf.size());

    // Parse HTTP request
    std::string method, path, body;
    std::stringstream ss(request);
    ss >> method >> path;

    auto header_end = request.find("\r\n\r\n");
    if (header_end != std::string::npos)
        body = request.substr(header_end + 4);

    // Handle CORS preflight
    if (method == "OPTIONS") {
        std::string empty_body;
        send_response(client_fd, 204, "text/plain", empty_body);
        return;
    }

    // Only handle GET and POST
    if (method != "GET" && method != "POST") {
        send_response(client_fd, 405, "application/json",
                       "{\"error\":\"method not allowed\"}");
        return;
    }

    std::string response_body;
    std::string content_type = "application/json";

    if (path == "/v1/completions" || path == "/completions") {
        if (method != "POST") {
            send_response(client_fd, 405, "application/json",
                           "{\"error\":\"POST required for completions\"}");
            return;
        }

        // Parse request body
        std::string prompt = "Hello";
        int max_tokens = 64;
        bool stream = false;

        // Look for stream flag
        if (body.find("\"stream\":true") != std::string::npos ||
            body.find("\"stream\": true") != std::string::npos) {
            stream = true;
        }

        // Parse prompt
        auto prompt_pos = body.find("\"prompt\"");
        if (prompt_pos != std::string::npos) {
            auto start_q = body.find('"', prompt_pos + 8);
            if (start_q != std::string::npos) {
                auto end_q = body.find('"', start_q + 1);
                if (end_q != std::string::npos)
                    prompt = body.substr(start_q + 1, end_q - start_q - 1);
            }
        }

        // Parse max_tokens
        auto max_pos = body.find("\"max_tokens\"");
        if (max_pos != std::string::npos) {
            auto colon = body.find(':', max_pos);
            if (colon != std::string::npos) {
                auto num_start = body.find_first_of("0123456789", colon + 1);
                if (num_start != std::string::npos) {
                    auto num_end = body.find_first_not_of("0123456789", num_start);
                    if (num_end == std::string::npos)
                        num_end = body.size();
                    max_tokens = std::stoi(body.substr(num_start, num_end - num_start));
                }
            }
        }

        max_tokens = (std::min)((std::max)(1, max_tokens), 1024);

        if (stream) {
            send_stream_response(client_fd, prompt, max_tokens);
            return;
        }

        if (model_) {
            BPETokenizer bpe;
            std::vector<int> tokens;
            try {
                tokens = bpe.encode(prompt);
            } catch (...) {
                send_response(client_fd, 500, "application/json",
                               "{\"error\":\"tokenization failed\"}");
                return;
            }
            if (tokens.empty()) tokens = {1};

            std::vector<int> output;
            output.reserve(max_tokens);

            for (int i = 0; i < max_tokens; i++) {
                try {
                    int64_t seq_len = (int64_t)(tokens.size() + output.size());
                    Tensor input({1, seq_len});
                    float* id = input.data<float>();
                    for (size_t j = 0; j < tokens.size(); j++) id[j] = (float)tokens[j];
                    for (size_t j = 0; j < output.size(); j++)
                        id[tokens.size() + j] = (float)output[j];
                    Tensor pos({1, seq_len});
                    Tensor logits = model_->forward(input, pos);
                    if (logits.numel() == 0) break;

                    int64_t V = logits.dim(logits.rank() - 1);
                    const float* ld = logits.data<float>() + logits.numel() - V;
                    int next = 0;
                    for (int64_t v = 1; v < V; v++)
                        if (ld[v] > ld[next]) next = (int)v;

                    if (next == 2) break;
                    output.push_back(next);
                } catch (...) {
                    break;
                }
            }

            std::string generated = bpe.decode(output);
            // Escape JSON string
            std::string escaped;
            for (char c : generated) {
                if (c == '"' || c == '\\') { escaped += '\\'; escaped += c; }
                else if (c == '\n') escaped += "\\n";
                else if (c == '\r') escaped += "\\r";
                else if (c == '\t') escaped += "\\t";
                else if ((unsigned char)c < 32) escaped += "";
                else escaped += c;
            }
            response_body = "{\"choices\":[{\"text\":\"" + escaped + "\"}]}";
        } else {
            response_body = "{\"error\":\"no model loaded\"}";
        }
        content_type = "application/json";
    } else if (path == "/v1/chat/completions") {
        if (method != "POST") {
            send_response(client_fd, 405, "application/json",
                           "{\"error\":\"POST required\"}");
            return;
        }
        // Simple passthrough to completions for now
        // Extract last message content as prompt
        std::string prompt = "Hello";
        auto content_pos = body.find("\"content\"");
        if (content_pos != std::string::npos) {
            auto start_q = body.find('"', content_pos + 9);
            if (start_q != std::string::npos) {
                auto end_q = body.find('"', start_q + 1);
                if (end_q != std::string::npos)
                    prompt = body.substr(start_q + 1, end_q - start_q - 1);
            }
        }
        response_body = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Echo: " + prompt + "\"}}]}";
        content_type = "application/json";
    } else if (path == "/health" || path == "/") {
        response_body = "{\"status\":\"ok\",\"model\":\"MYTHOS.cpp\"}";
        content_type = "application/json";
    } else if (path == "/v1/models") {
        if (!model_) {
            response_body = "{\"data\":[]}";
        } else {
            std::string model_name = model_->config.vocab_size > 0 ? "default" : "unknown";
            response_body = "{\"data\":[{\"id\":\"" + model_name + "\",\"object\":\"model\"}]}";
        }
        content_type = "application/json";
    } else {
        send_response(client_fd, 404, "application/json",
                       "{\"error\":\"not found\",\"path\":\"" + path + "\"}");
        return;
    }

    send_response(client_fd, 200, content_type, response_body);
}

// ========================================================================
// I6: WebSocket
// ========================================================================
WebSocketHandler::WebSocketHandler(int port) : port_(port) {}

WebSocketHandler::~WebSocketHandler() {
    stop();
}

void WebSocketHandler::start() {
    if (running_) return;
    running_ = true;
    accept_thread_ = std::thread(&WebSocketHandler::accept_loop, this);
}

void WebSocketHandler::stop() {
    running_ = false;
    // Close all client connections
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (int fd : clients_) {
            close_socket_impl(fd);
        }
        clients_.clear();
    }
    if (accept_thread_.joinable()) accept_thread_.join();
    for (auto& t : client_threads_) {
        if (t.joinable()) t.join();
    }
    client_threads_.clear();
}

void WebSocketHandler::remove_client(int client_fd) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    auto it = std::find(clients_.begin(), clients_.end(), client_fd);
    if (it != clients_.end()) {
        close_socket_impl(*it);
        clients_.erase(it);
    }
}

void WebSocketHandler::close_socket(int fd) {
    close_socket_impl(fd);
}

// Compute Sec-WebSocket-Accept value
std::string WebSocketHandler::compute_accept_key(const std::string& client_key) {
    static const char* magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string concat = client_key + magic;
    uint8_t hash[20];
    sha1((const uint8_t*)concat.data(), concat.size(), hash);
    return base64_encode(hash, 20);
}

std::string WebSocketHandler::base64_encode(const uint8_t* data, size_t len) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                "abcdefghijklmnopqrstuvwxyz"
                                "0123456789+/";
    std::string out;
    out.reserve((len + 2) / 3 * 4);

    for (size_t i = 0; i < len; i += 3) {
        uint32_t val = (uint32_t)data[i] << 16;
        if (i + 1 < len) val |= (uint32_t)data[i+1] << 8;
        if (i + 2 < len) val |= (uint32_t)data[i+2];

        out += table[(val >> 18) & 0x3F];
        out += table[(val >> 12) & 0x3F];
        out += (i + 1 < len) ? table[(val >> 6) & 0x3F] : '=';
        out += (i + 2 < len) ? table[val & 0x3F] : '=';
    }
    return out;
}

void WebSocketHandler::sha1(const uint8_t* data, size_t len, uint8_t out[20]) {
    SHA1_CTX ctx;
    SHA1_CTX::init(&ctx);
    SHA1_CTX::update(&ctx, data, len);
    SHA1_CTX::final(&ctx, out);
}

std::vector<uint8_t> WebSocketHandler::create_frame(const std::string& data,
                                                      uint8_t opcode) {
    size_t len = data.size();
    std::vector<uint8_t> frame;

    // FIN + opcode
    frame.push_back((uint8_t)(0x80 | opcode));

    // Payload length
    if (len < 126) {
        frame.push_back((uint8_t)len);
    } else if (len < 65536) {
        frame.push_back(126);
        frame.push_back((uint8_t)(len >> 8));
        frame.push_back((uint8_t)(len & 0xFF));
    } else {
        frame.push_back(127);
        for (int i = 7; i >= 0; i--)
            frame.push_back((uint8_t)(len >> (i * 8)));
    }

    // Payload
    frame.insert(frame.end(), data.begin(), data.end());
    return frame;
}

bool WebSocketHandler::parse_frame(const uint8_t* buf, size_t len,
                                    std::string& payload, uint8_t& opcode,
                                    bool& fin) {
    if (len < 2) return false;

    fin = (buf[0] & 0x80) != 0;
    opcode = buf[0] & 0x0F;
    bool masked = (buf[1] & 0x80) != 0;
    uint64_t payload_len = buf[1] & 0x7F;

    size_t offset = 2;

    if (payload_len == 126) {
        if (len < 4) return false;
        payload_len = ((uint64_t)buf[2] << 8) | buf[3];
        offset = 4;
    } else if (payload_len == 127) {
        if (len < 10) return false;
        payload_len = 0;
        for (int i = 0; i < 8; i++)
            payload_len = (payload_len << 8) | buf[2 + i];
        offset = 10;
    }

    uint8_t mask_key[4] = {0, 0, 0, 0};
    if (masked) {
        if (len < offset + 4) return false;
        memcpy(mask_key, buf + offset, 4);
        offset += 4;
    }

    if (len < offset + payload_len) return false;

    payload.assign((const char*)(buf + offset), (size_t)payload_len);

    // Unmask if needed
    if (masked) {
        for (size_t i = 0; i < payload.size(); i++)
            payload[i] ^= mask_key[i & 3];
    }

    return true;
}

void WebSocketHandler::accept_loop() {
    if (!socket_platform_init()) return;

#ifdef _WIN32
    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) { socket_platform_cleanup(); return; }
#else
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { socket_platform_cleanup(); return; }
#endif

    int opt = 1;
#ifdef _WIN32
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR,
               (const char*)&opt, sizeof(opt));
#else
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((u_short)port_);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        closesocket(server_fd);
        socket_platform_cleanup();
        return;
    }

    if (listen(server_fd, 16) < 0) {
        closesocket(server_fd);
        socket_platform_cleanup();
        return;
    }

    fd_set read_fds;
    while (running_) {
        FD_ZERO(&read_fds);
#ifdef _WIN32
        FD_SET(server_fd, &read_fds);
#else
        FD_SET(server_fd, &read_fds);
#endif
        struct timeval tv = {1, 0};

        if (select((int)server_fd + 1, &read_fds, nullptr, nullptr, &tv) > 0) {
            sockaddr_in client_addr;
#ifdef _WIN32
            int addr_len = sizeof(client_addr);
#else
            socklen_t addr_len = sizeof(client_addr);
#endif
            int client_fd = (int)accept(server_fd, (sockaddr*)&client_addr, &addr_len);
            if (client_fd < 0) continue;

            // Read WebSocket upgrade request
            char buf[4096];
            int n = recv(client_fd, buf, sizeof(buf) - 1, 0);
            if (n <= 0) {
                close_socket_impl(client_fd);
                continue;
            }
            buf[n] = '\0';
            std::string req(buf);

            if (req.find("Upgrade: websocket") == std::string::npos &&
                req.find("upgrade: websocket") == std::string::npos) {
                // Not a WebSocket request
                std::string resp = "HTTP/1.1 400 Bad Request\r\n\r\n";
                send(client_fd, resp.c_str(), (int)resp.size(), 0);
                close_socket_impl(client_fd);
                continue;
            }

            // Extract WebSocket key
            std::string ws_key;
            auto key_pos = req.find("Sec-WebSocket-Key:");
            if (key_pos == std::string::npos)
                key_pos = req.find("sec-websocket-key:");
            if (key_pos != std::string::npos) {
                auto val_start = req.find_first_not_of(" \t", key_pos + 18);
                if (val_start != std::string::npos) {
                    auto val_end = req.find("\r\n", val_start);
                    if (val_end != std::string::npos)
                        ws_key = req.substr(val_start, val_end - val_start);
                }
            }

            if (ws_key.empty()) {
                ws_key = "dGhlIHNhbXBsZSBub25jZQ==";
            }

            std::string accept = compute_accept_key(ws_key);

            std::string resp =
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Accept: " + accept + "\r\n"
                "\r\n";

            send(client_fd, resp.c_str(), (int)resp.size(), 0);

            {
                std::lock_guard<std::mutex> lock(clients_mutex_);
                clients_.push_back(client_fd);
            }

            // Spawn client handler thread
            client_threads_.emplace_back(&WebSocketHandler::client_loop, this, client_fd);
        }
    }

    closesocket(server_fd);
    socket_platform_cleanup();
}

void WebSocketHandler::client_loop(int client_fd) {
    set_socket_timeout_impl(client_fd, 30);
    uint8_t buf[4096];

    while (running_) {
#ifdef _WIN32
        int n = recv(client_fd, (char*)buf, sizeof(buf), 0);
#else
        int n = recv(client_fd, (char*)buf, sizeof(buf), 0);
#endif
        if (n <= 0) {
            // Client disconnected
            break;
        }

        std::string payload;
        uint8_t opcode = 0;
        bool fin = false;

        if (!parse_frame(buf, (size_t)n, payload, opcode, fin)) {
            break;
        }

        switch (opcode) {
            case 0x8: // Close frame
                {
                    auto close_frame = create_frame("", 0x8);
                    send(client_fd, (const char*)close_frame.data(),
                         (int)close_frame.size(), 0);
                }
                goto disconnect;

            case 0x9: // Ping
                {
                    auto pong = create_frame(payload, 0xA);
                    send(client_fd, (const char*)pong.data(),
                         (int)pong.size(), 0);
                }
                break;

            case 0xA: // Pong
                break;

            case 0x1: // Text frame
            case 0x2: // Binary frame
                // Echo back for now
                {
                    auto echo = create_frame(payload, opcode);
                    send(client_fd, (const char*)echo.data(),
                         (int)echo.size(), 0);
                }
                break;
        }
    }

disconnect:
    remove_client(client_fd);
}

void WebSocketHandler::broadcast(const std::string& msg) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    auto frame = create_frame(msg, 0x1); // Text opcode

    auto it = clients_.begin();
    while (it != clients_.end()) {
        int client = *it;
        int sent = send(client, (const char*)frame.data(),
                        (int)frame.size(), 0);
        if (sent < 0) {
            close_socket_impl(client);
            it = clients_.erase(it);
        } else {
            ++it;
        }
    }
}

// ========================================================================
// I12: Logger
// ========================================================================
Logger::Logger(Level level) : level_(level) {}

std::string Logger::level_str(Level l) {
    switch (l) {
        case DEBUG: return "DEBUG";
        case INFO:  return "INFO";
        case WARN:  return "WARN";
        case ERROR: return "ERROR";
        default:    return "UNKNOWN";
    }
}

void Logger::log(Level level, const std::string& message) {
    if (level < level_) return;
    std::lock_guard<std::mutex> lock(mtx_);
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    std::string line = std::string(buf) + " [" + level_str(level) + "] " + message;
    if (!file_path_.empty()) {
        std::ofstream f(file_path_, std::ios::app);
        if (f) f << line << std::endl;
    }
    std::cout << line << std::endl;
}

void Logger::set_file(const std::string& path) { file_path_ = path; }
Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

// ========================================================================
// I13: Config (JSON parser)
// ========================================================================
AppConfig::AppConfig(const std::string& path) {
    if (path.empty()) return;

    std::ifstream f(path);
    if (!f) return;

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    if (content.empty()) return;

    std::string error;
    size_t pos = 0;
    root_ = parse_json(content, pos, &error);

    if (!error.empty()) {
        Logger::instance().log(Logger::WARN,
            "Config parse error in " + path + ": " + error + " at position " +
            std::to_string(pos));
    }
}

void AppConfig::skip_ws(const std::string& json, size_t& pos) {
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
           json[pos] == '\n' || json[pos] == '\r')) {
        pos++;
    }
}

std::string AppConfig::escape_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (size_t i = 0; i < s.size(); i++) {
        char c = s[i];
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

AppConfig::JsonValue AppConfig::auto_type_value(const std::string& val) {
    // Try parsing as integer
    char* end = nullptr;
    long long ll = strtoll(val.c_str(), &end, 10);
    if (end && *end == '\0' && end != val.c_str()) {
        return JsonValue((int64_t)ll);
    }

    // Try as float
    end = nullptr;
    double d = strtod(val.c_str(), &end);
    if (end && *end == '\0' && end != val.c_str()) {
        return JsonValue(d);
    }

    // Handle booleans
    if (val == "true") return JsonValue(true);
    if (val == "false") return JsonValue(false);

    // Fallback to string
    return JsonValue(val);
}

std::string AppConfig::parse_string(const std::string& json, size_t& pos,
                                     std::string* error) {
    std::string result;
    if (pos >= json.size() || json[pos] != '"') {
        if (error) *error = "expected string";
        return result;
    }
    pos++; // skip opening quote

    while (pos < json.size()) {
        char c = json[pos];
        if (c == '"') {
            pos++;
            return result;
        }
        if (c == '\\') {
            pos++;
            if (pos >= json.size()) break;
            switch (json[pos]) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case '/':  result += '/'; break;
                case 'b':  result += '\b'; break;
                case 'f':  result += '\f'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                case 'u': {
                    // Parse 4 hex digits
                    if (pos + 4 < json.size()) {
                        // Simplified: just grab the raw hex digits
                        result += "\\u";
                        for (int i = 0; i < 4; i++) {
                            pos++;
                            result += json[pos];
                        }
                    }
                    break;
                }
                default: result += json[pos]; break;
            }
            pos++;
        } else {
            result += c;
            pos++;
        }
    }

    if (error) *error = "unterminated string";
    return result;
}

AppConfig::JsonValue AppConfig::parse_number(const std::string& json,
                                              size_t& pos, std::string* error) {
    size_t start = pos;
    if (pos < json.size() && json[pos] == '-') pos++;

    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') pos++;

    bool is_float = false;
    if (pos < json.size() && json[pos] == '.') {
        is_float = true;
        pos++;
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') pos++;
    }

    if (pos < json.size() && (json[pos] == 'e' || json[pos] == 'E')) {
        is_float = true;
        pos++;
        if (pos < json.size() && (json[pos] == '+' || json[pos] == '-')) pos++;
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') pos++;
    }

    std::string num_str = json.substr(start, pos - start);

    if (is_float) {
        char* end = nullptr;
        double d = strtod(num_str.c_str(), &end);
        (void)end;
        return JsonValue(d);
    } else {
        char* end = nullptr;
        long long ll = strtoll(num_str.c_str(), &end, 10);
        (void)end;
        return JsonValue((int64_t)ll);
    }
}

AppConfig::JsonValue AppConfig::parse_value(const std::string& json,
                                             size_t& pos, std::string* error) {
    skip_ws(json, pos);
    if (pos >= json.size()) {
        if (error) *error = "unexpected end of JSON";
        return JsonValue();
    }

    switch (json[pos]) {
        case '{': return parse_object(json, pos, error);
        case '[': return parse_array(json, pos, error);
        case '"': return JsonValue(parse_string(json, pos, error));
        case 't':
            if (json.substr(pos, 4) == "true") { pos += 4; return JsonValue(true); }
            if (error) *error = "expected true";
            return JsonValue();
        case 'f':
            if (json.substr(pos, 5) == "false") { pos += 5; return JsonValue(false); }
            if (error) *error = "expected false";
            return JsonValue();
        case 'n':
            if (json.substr(pos, 4) == "null") { pos += 4; return JsonValue(); }
            if (error) *error = "expected null";
            return JsonValue();
        default:
            if (json[pos] == '-' || (json[pos] >= '0' && json[pos] <= '9'))
                return parse_number(json, pos, error);
            if (error) *error = std::string("unexpected character '") + json[pos] + "'";
            return JsonValue();
    }
}

AppConfig::JsonValue AppConfig::parse_object(const std::string& json,
                                              size_t& pos, std::string* error) {
    std::unordered_map<std::string, JsonValue> obj;
    pos++; // skip '{'

    skip_ws(json, pos);
    if (pos < json.size() && json[pos] == '}') {
        pos++;
        return JsonValue(obj);
    }

    while (pos < json.size()) {
        skip_ws(json, pos);
        if (pos >= json.size()) {
            if (error) *error = "unterminated object";
            break;
        }

        if (json[pos] != '"') {
            if (error) *error = "expected string key in object";
            break;
        }

        std::string key = parse_string(json, pos, error);

        skip_ws(json, pos);
        if (pos >= json.size() || json[pos] != ':') {
            if (error) *error = "expected ':' in object";
            break;
        }
        pos++; // skip ':'

        JsonValue val = parse_value(json, pos, error);
        obj[key] = std::move(val);

        skip_ws(json, pos);
        if (pos >= json.size()) break;

        if (json[pos] == '}') {
            pos++;
            return JsonValue(std::move(obj));
        }

        if (json[pos] != ',') {
            if (error) *error = "expected ',' or '}' in object";
            break;
        }
        pos++; // skip ','
    }

    return JsonValue(std::move(obj));
}

AppConfig::JsonValue AppConfig::parse_array(const std::string& json,
                                             size_t& pos, std::string* error) {
    std::vector<JsonValue> arr;
    pos++; // skip '['

    skip_ws(json, pos);
    if (pos < json.size() && json[pos] == ']') {
        pos++;
        return JsonValue(std::move(arr));
    }

    while (pos < json.size()) {
        arr.push_back(parse_value(json, pos, error));

        skip_ws(json, pos);
        if (pos >= json.size()) break;

        if (json[pos] == ']') {
            pos++;
            return JsonValue(std::move(arr));
        }

        if (json[pos] != ',') {
            if (error) *error = "expected ',' or ']' in array";
            break;
        }
        pos++; // skip ','
    }

    return JsonValue(std::move(arr));
}

AppConfig::JsonValue AppConfig::parse_json(const std::string& json,
                                            size_t& pos, std::string* error) {
    return parse_value(json, pos, error);
}

void AppConfig::serialize_json(const JsonValue& val, std::string& out, int indent) {
    std::string ind(indent, ' ');
    std::string ind_inner(indent + 2, ' ');

    switch (val.type) {
        case JsonValue::NULL_VAL:
            out += "null";
            break;
        case JsonValue::BOOL:
            out += val.bool_val ? "true" : "false";
            break;
        case JsonValue::INT64:
            out += std::to_string(val.int_val);
            break;
        case JsonValue::FLOAT64: {
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", val.float_val);
            out += buf;
            break;
        }
        case JsonValue::STRING:
            out += "\"" + escape_string(val.str_val) + "\"";
            break;
        case JsonValue::ARRAY:
            out += "[";
            for (size_t i = 0; i < val.arr_val.size(); i++) {
                if (i > 0) out += ",";
                serialize_json(val.arr_val[i], out, indent + 2);
            }
            out += "]";
            break;
        case JsonValue::OBJECT:
            out += "{";
            if (!val.obj_val.empty()) {
                bool first = true;
                for (auto& [k, v] : val.obj_val) {
                    if (!first) out += ",";
                    first = false;
                    out += "\"" + escape_string(k) + "\":";
                    serialize_json(v, out, indent + 2);
                }
            }
            out += "}";
            break;
    }
}

AppConfig::JsonValue* AppConfig::resolve_path(const std::string& key) {
    return const_cast<JsonValue*>(const_cast<const AppConfig*>(this)->resolve_path(key));
}

const AppConfig::JsonValue* AppConfig::resolve_path(const std::string& key) const {
    const JsonValue* current = &root_;

    // Split by dots
    size_t start = 0;
    while (start < key.size()) {
        auto dot = key.find('.', start);
        std::string part = key.substr(start, dot - start);
        start = (dot == std::string::npos) ? key.size() : dot + 1;

        if (current->type != JsonValue::OBJECT) return nullptr;
        auto it = current->obj_val.find(part);
        if (it == current->obj_val.end()) return nullptr;
        current = &it->second;
    }

    return current;
}

float AppConfig::get_float(const std::string& key, float def) const {
    const JsonValue* val = resolve_path(key);
    if (!val) return def;

    switch (val->type) {
        case JsonValue::FLOAT64: return (float)val->float_val;
        case JsonValue::INT64:   return (float)val->int_val;
        case JsonValue::STRING:
            try { return std::stof(val->str_val); } catch (...) { return def; }
        case JsonValue::BOOL:    return val->bool_val ? 1.0f : 0.0f;
        default: return def;
    }
}

int AppConfig::get_int(const std::string& key, int def) const {
    const JsonValue* val = resolve_path(key);
    if (!val) return def;

    switch (val->type) {
        case JsonValue::INT64:   return (int)val->int_val;
        case JsonValue::FLOAT64: return (int)val->float_val;
        case JsonValue::STRING:
            try { return std::stoi(val->str_val); } catch (...) { return def; }
        case JsonValue::BOOL:    return val->bool_val ? 1 : 0;
        default: return def;
    }
}

std::string AppConfig::get_string(const std::string& key,
                                   const std::string& def) const {
    const JsonValue* val = resolve_path(key);
    if (!val) return def;

    switch (val->type) {
        case JsonValue::STRING: return val->str_val;
        case JsonValue::INT64:  return std::to_string(val->int_val);
        case JsonValue::FLOAT64: return std::to_string(val->float_val);
        case JsonValue::BOOL:   return val->bool_val ? "true" : "false";
        case JsonValue::NULL_VAL: return "null";
        default: return def;
    }
}

void AppConfig::set(const std::string& key, const std::string& value) {
    if (root_.type != JsonValue::OBJECT && root_.type != JsonValue::NULL_VAL) {
        root_ = JsonValue(std::unordered_map<std::string, JsonValue>());
    }
    if (root_.type == JsonValue::NULL_VAL) {
        root_.type = JsonValue::OBJECT;
    }

    JsonValue* current = &root_;

    // Split by dots, creating intermediate objects
    size_t start = 0;
    while (start < key.size()) {
        auto dot = key.find('.', start);
        std::string part = key.substr(start, dot - start);
        start = (dot == std::string::npos) ? key.size() : dot + 1;

        if (start >= key.size()) {
            // Last component — set the value
            current->obj_val[part] = auto_type_value(value);
        } else {
            // Intermediate — ensure object exists
            auto it = current->obj_val.find(part);
            if (it == current->obj_val.end() ||
                it->second.type != JsonValue::OBJECT) {
                current->obj_val[part] = JsonValue(
                    std::unordered_map<std::string, JsonValue>());
            }
            current = &current->obj_val[part];
        }
    }
}

void AppConfig::save(const std::string& path) {
    std::string json = to_json();
    std::ofstream f(path);
    if (f) f << json << std::endl;
}

std::string AppConfig::to_json() const {
    std::string out;
    serialize_json(root_, out, 0);
    return out;
}

bool AppConfig::validate(std::string* error_out) const {
    std::string error;
    size_t pos = 0;
    std::string json = to_json();
    JsonValue test = parse_json(json, pos, &error);
    if (!error.empty()) {
        if (error_out) *error_out = error;
        return false;
    }
    return pos >= json.size();
}

// ========================================================================
// I14: Plugin system
// ========================================================================
PluginManager::~PluginManager() {
    unload_all();
}

Plugin* PluginManager::load_from_dll(const std::string& path) {
#ifdef _WIN32
    HMODULE h = LoadLibraryA(path.c_str());
    if (!h) return nullptr;

    using CreatePluginFn = Plugin*(*)();
    auto create_fn = (CreatePluginFn)GetProcAddress(h, "create_plugin");
    if (!create_fn) {
        FreeLibrary(h);
        return nullptr;
    }

    Plugin* p = create_fn();
    if (!p) {
        FreeLibrary(h);
        return nullptr;
    }

    PluginEntry entry;
    entry.plugin = p;
    entry.path = path;
    entry.name = p->name();
    entry.handle = h;
    entries_.push_back(entry);
    return p;
#else
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) return nullptr;

    using CreatePluginFn = Plugin*(*)();
    auto create_fn = (CreatePluginFn)dlsym(handle, "create_plugin");
    if (!create_fn) {
        dlclose(handle);
        return nullptr;
    }

    Plugin* p = create_fn();
    if (!p) {
        dlclose(handle);
        return nullptr;
    }

    PluginEntry entry;
    entry.plugin = p;
    entry.path = path;
    entry.name = p->name();
    entry.handle = handle;
    entries_.push_back(entry);
    return p;
#endif
}

void PluginManager::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);

    // Check if already loaded
    for (auto& e : entries_) {
        if (e.path == path) return;
    }

    Plugin* p = load_from_dll(path);
    if (!p) {
        Logger::instance().log(Logger::ERROR,
            "Failed to load plugin: " + path);
        PluginEntry failed_entry;
        failed_entry.path = path;
        failed_entry.load_error = "load failed";
        entries_.push_back(failed_entry);
    } else {
        Logger::instance().log(Logger::INFO,
            "Loaded plugin: " + p->name() + " from " + path);
    }
}

void PluginManager::unload(const std::string& name) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);

    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->name == name) {
            if (it->plugin) {
                delete it->plugin;
            }
#ifdef _WIN32
            if (it->handle) FreeLibrary((HMODULE)it->handle);
#else
            if (it->handle) dlclose(it->handle);
#endif
            entries_.erase(it);
            return;
        }
    }
}

void PluginManager::unload_all() {
    std::lock_guard<std::mutex> lock(plugins_mutex_);

    for (auto& e : entries_) {
        if (e.plugin) delete e.plugin;
#ifdef _WIN32
        if (e.handle) FreeLibrary((HMODULE)e.handle);
#else
        if (e.handle) dlclose(e.handle);
#endif
    }
    entries_.clear();

    direct_plugins_.clear();
}

bool PluginManager::hot_reload(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);

    // Find existing plugin by path
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->path == path) {
            // Unload old plugin
            std::string name = it->name;
            if (it->plugin) delete it->plugin;
#ifdef _WIN32
            if (it->handle) FreeLibrary((HMODULE)it->handle);
#else
            if (it->handle) dlclose(it->handle);
#endif
            entries_.erase(it);

            // Reload
            Plugin* p = load_from_dll(path);
            if (p) {
                Logger::instance().log(Logger::INFO,
                    "Hot-reloaded plugin: " + p->name());
                return true;
            }
            Logger::instance().log(Logger::ERROR,
                "Hot-reload failed for: " + path);
            return false;
        }
    }

    // Not found, try loading
    load(path);
    return true;
}

void PluginManager::register_plugin(Plugin* p) {
    if (!p) return;
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    direct_plugins_.push_back(p);
}

void PluginManager::on_generate_start(const std::string& prompt) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    for (auto& e : entries_) {
        if (e.plugin) e.plugin->on_generate_start(prompt);
    }
    for (auto* p : direct_plugins_) {
        p->on_generate_start(prompt);
    }
}

void PluginManager::on_token_generated(int token) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    for (auto& e : entries_) {
        if (e.plugin) e.plugin->on_token_generated(token);
    }
    for (auto* p : direct_plugins_) {
        p->on_token_generated(token);
    }
}

void PluginManager::on_generate_end(const std::string& output) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    for (auto& e : entries_) {
        if (e.plugin) e.plugin->on_generate_end(output);
    }
    for (auto* p : direct_plugins_) {
        p->on_generate_end(output);
    }
}

// ========================================================================
// I15: Model zoo
// ========================================================================
ModelZoo::ModelZoo(const std::string& zoo_path)
    : zoo_path_(zoo_path) {
    // Ensure path ends with separator
    if (!zoo_path_.empty()) {
#ifdef _WIN32
        if (zoo_path_.back() != '\\' && zoo_path_.back() != '/')
            zoo_path_ += '\\';
#else
        if (zoo_path_.back() != '/')
            zoo_path_ += '/';
#endif
    }
}

void ModelZoo::scan_directory(std::vector<ModelInfo>& out) const {
#ifdef _WIN32
    std::string search = zoo_path_ + "*.oil";
    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(search.c_str(), &findData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
            std::string name = findData.cFileName;
            std::string full_path = zoo_path_ + name;

            // Try to load and read header for validation
            DenseModel m;
            int64_t params = 0;
            try {
                m.load(full_path);
                params = m.param_count();
            } catch (...) {
                continue;
            }

            out.push_back({
                name.substr(0, name.find_last_of('.')),
                full_path,
                params,
                "OIL8"
            });
        } while (FindNextFileA(hFind, &findData) != 0);
        FindClose(hFind);
    }
#else
    DIR* dir = opendir(zoo_path_.c_str());
    if (!dir) return;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;

        // Check extension
        if (name.size() < 4) continue;
        std::string ext = name.substr(name.size() - 4);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        if (ext != ".oil") continue;

        std::string full_path = zoo_path_ + name;

        // Validate by trying to load
        DenseModel m;
        int64_t params = 0;
        try {
            m.load(full_path);
            params = m.param_count();
        } catch (...) {
            continue;
        }

        out.push_back({
            name.substr(0, name.size() - 4),
            full_path,
            params,
            "OIL8"
        });
    }
    closedir(dir);
#endif
}

std::vector<ModelZoo::ModelInfo> ModelZoo::list_models() const {
    if (cache_valid_) return cache_;

    cache_.clear();
    scan_directory(cache_);

    // If no models found, add defaults
    if (cache_.empty()) {
        cache_.push_back({"tiny", zoo_path_ + "tiny.oil", 85000000, "OIL8"});
        cache_.push_back({"small", zoo_path_ + "small.oil", 350000000, "OIL8"});
    }

    cache_valid_ = true;
    return cache_;
}

Model* ModelZoo::load(const std::string& name) {
    // Check cache
    if (!cache_valid_) list_models();

    for (auto& m : cache_) {
        if (m.name == name || m.path.find(name) != std::string::npos) {
            auto* model = new DenseModel();
            try {
                model->load(m.path);
                return model;
            } catch (...) {
                delete model;
                return nullptr;
            }
        }
    }

    // Try direct path
    auto* model = new DenseModel();
    try {
        model->load(name);
        return model;
    } catch (...) {
        delete model;
    }

    // Try zoo_path + name
    std::string direct_path = zoo_path_ + name;
#ifdef _WIN32
    direct_path += ".oil";
#endif
    model = new DenseModel();
    try {
        model->load(direct_path);
        cache_.push_back({name, direct_path, model->param_count(), "OIL8"});
        return model;
    } catch (...) {
        delete model;
    }

    return nullptr;
}

// ========================================================================
// I16-I18: Language bindings
// ========================================================================
void PythonBindings::init() {
    Logger::instance().log(Logger::INFO,
        "Python bindings stub — requires pybind11 in separate .cpp");
}

void JavaBindings::init() {
    Logger::instance().log(Logger::INFO,
        "Java bindings stub — requires jni.h in separate .cpp");
}

void RustBindings::init() {
    Logger::instance().log(Logger::INFO,
        "Rust bindings stub — extern \"C\" functions with C ABI");
}

// ========================================================================
// I19-I20: Mobile/WASM
// ========================================================================
bool MobileDeploy::deploy_android(const std::string& apk_path) {
    Logger::instance().log(Logger::INFO,
        "Android deploy stub for: " + apk_path);
    return false;
}

bool MobileDeploy::deploy_ios(const std::string& xcarchive_path) {
    Logger::instance().log(Logger::INFO,
        "iOS deploy stub for: " + xcarchive_path);
    return false;
}

bool WASMDeploy::compile_to_wasm(const std::string& source_path) {
    Logger::instance().log(Logger::INFO,
        "WASM compile stub for: " + source_path);
    return false;
}

} // namespace oil
