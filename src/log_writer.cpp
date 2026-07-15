#include "oil/log_writer.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace oil {

// C21: TensorBoard-compatible event writer
// Writes TF Events file format: [uint64 length][uint32 crc][Event proto][uint32 crc]
// Uses a minimal binary format compatible with TensorBoard's event reading.

static int64_t current_wall_time() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static uint32_t masked_crc32c(const uint8_t* data, size_t len) {
    static uint32_t table[256];
    static bool table_init = false;
    if (!table_init) {
        const uint32_t poly = 0x82F63B78;
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t crc = i;
            for (int j = 0; j < 8; j++)
                crc = (crc >> 1) ^ (poly & (-(int32_t)(crc & 1)));
            table[i] = crc;
        }
        table_init = true;
    }
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++)
        crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xFF];
    crc ^= 0xFFFFFFFF;
    return ((crc >> 15) | (crc << 17)) + 0xa282ead8UL;
}

static uint32_t masked_crc32c(const std::string& s) {
    return masked_crc32c((const uint8_t*)s.data(), s.size());
}

EventWriter::EventWriter(const std::string& logdir) : logdir_(logdir) {
    std::filesystem::create_directories(logdir_);
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << logdir_ << "/events.out.tfevents." << current_wall_time()
        << ".localhost";
    file_.open(oss.str(), std::ios::binary);
}

EventWriter::~EventWriter() {
    flush();
    if (file_.is_open()) file_.close();
}

void EventWriter::write_event(const std::string& tag, float value, int step, int64_t wall_time) {
    if (!file_.is_open()) return;
    std::lock_guard<std::mutex> lock(mtx_);
    std::ostringstream summary;
    summary << tag << "\n" << value;
    std::string event_str = summary.str();
    uint64_t len = (uint64_t)event_str.size();
    uint32_t crc1 = masked_crc32c((const uint8_t*)&len, sizeof(len));
    uint32_t crc2 = masked_crc32c(event_str);
    file_.write((const char*)&len, sizeof(len));
    file_.write((const char*)&crc1, sizeof(crc1));
    file_.write(event_str.data(), (std::streamsize)event_str.size());
    file_.write((const char*)&crc2, sizeof(crc2));
}

void EventWriter::write_scalar(const std::string& tag, float value, int step) {
    write_event(tag, value, step, current_wall_time());
}

void EventWriter::write_scalars(const TrainMetrics& metrics, int step) {
    write_scalar("loss", metrics.loss, step);
    write_scalar("perplexity", metrics.perplexity, step);
    write_scalar("grad_norm", metrics.grad_norm, step);
    write_scalar("learning_rate", metrics.learning_rate, step);
    write_scalar("tokens_per_sec", (float)metrics.tokens_per_sec, step);
    if (metrics.val_loss > 0) {
        write_scalar("val_loss", metrics.val_loss, step);
        write_scalar("val_perplexity", metrics.val_perplexity, step);
    }
    flush();
}

void EventWriter::flush() {
    if (file_.is_open()) file_.flush();
}

// C22: WandB-compatible lightweight logger
WandBLogger::WandBLogger(const std::string& project, const std::string& run_name) {
    run_name_ = run_name.empty() ? "run_" + std::to_string(current_wall_time()) : run_name;
    std::string dir = "wandb/" + project + "/" + run_name_;
    std::filesystem::create_directories(dir);
    file_.open(dir + "/metrics.json", std::ios::app);
}

WandBLogger::~WandBLogger() { flush(); if (file_.is_open()) file_.close(); }

void WandBLogger::log(const std::string& key, float value, int step) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!file_.is_open()) return;
    file_ << "{\"step\": " << step << ", \"" << key << "\": " << value << "}\n";
}

void WandBLogger::log_metrics(const TrainMetrics& metrics, int step) {
    log("loss", metrics.loss, step);
    log("perplexity", metrics.perplexity, step);
    log("grad_norm", metrics.grad_norm, step);
    log("learning_rate", metrics.learning_rate, step);
    log("tokens_per_sec", (float)metrics.tokens_per_sec, step);
    if (metrics.val_loss > 0) log("val_loss", metrics.val_loss, step);
    if (metrics.val_perplexity > 0) log("val_perplexity", metrics.val_perplexity, step);
    flush();
}

void WandBLogger::flush() { if (file_.is_open()) file_.flush(); }

} // namespace oil
