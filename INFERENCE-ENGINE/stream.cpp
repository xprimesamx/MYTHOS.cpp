#include "inference.h"
#include <cctype>

namespace oil {
namespace engine {

struct StreamBuffer {
    std::string buffer;
    void flush(std::function<void(const std::string&)> on_token);
    void append(const std::string& token);
    bool should_flush() const;
};

void StreamBuffer::flush(std::function<void(const std::string&)> on_token) {
    if (!buffer.empty()) {
        on_token(buffer);
        buffer.clear();
    }
}

void StreamBuffer::append(const std::string& token) {
    buffer += token;
}

bool StreamBuffer::should_flush() const {
    if (buffer.empty()) return false;
    unsigned char c = static_cast<unsigned char>(buffer.back());
    if (c == ' ' || c == '\n' || c == '\t') return true;
    if (c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':') return true;
    if (c == ')' || c == '}' || c == ']') return true;
    if (c == '\"') return true;
    if (c == '\'') return true;
    if (c == '\r') return true;
    return false;
}

} // namespace engine
} // namespace oil
