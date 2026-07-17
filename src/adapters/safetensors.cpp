#include "oil/adapters.h"

#include <cstring>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace mythos {
namespace adapters {

SafetensorsImporter::SafetensorsImporter() {}

static std::string read_json_string(const std::string& json, size_t& pos) {
    while (pos < json.size() && json[pos] != '"') pos++;
    pos++;
    size_t start = pos;
    while (pos < json.size() && json[pos] != '"') pos++;
    return json.substr(start, pos - start);
}

static int64_t parse_int(const std::string& s) {
    int64_t val = 0;
    for (char c : s) {
        if (c >= '0' && c <= '9') val = val * 10 + (c - '0');
    }
    return val;
}

bool SafetensorsImporter::read(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    uint64_t header_size;
    f.read(reinterpret_cast<char*>(&header_size), 8);
    if (header_size > 100 * 1024 * 1024) return false;

    std::string header_json(header_size, '\0');
    f.read(&header_json[0], header_size);

    size_t pos = 0;
    while (pos < header_json.size()) {
        size_t key_start = header_json.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = header_json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = header_json.substr(key_start + 1, key_end - key_start - 1);

        if (key == "__metadata__") {
            pos = header_json.find('}', key_end);
            if (pos == std::string::npos) break;
            pos++;
            continue;
        }

        size_t obj_start = header_json.find('{', key_end);
        size_t obj_end = header_json.find('}', obj_start);
        if (obj_start == std::string::npos || obj_end == std::string::npos) break;

        std::string obj = header_json.substr(obj_start, obj_end - obj_start + 1);

        TensorRecord rec;
        rec.name = key;

        size_t dtype_pos = obj.find("\"dtype\"");
        if (dtype_pos != std::string::npos) {
            size_t dt_val = obj.find('"', dtype_pos + 8);
            size_t dt_end = obj.find('"', dt_val + 1);
            std::string dtype = obj.substr(dt_val + 1, dt_end - dt_val - 1);
        }

        size_t shape_pos = obj.find("\"shape\"");
        if (shape_pos != std::string::npos) {
            size_t arr_start = obj.find('[', shape_pos);
            size_t arr_end = obj.find(']', arr_start);
            std::string shape_str = obj.substr(arr_start + 1, arr_end - arr_start - 1);
            std::istringstream ss(shape_str);
            std::string dim;
            while (std::getline(ss, dim, ',')) {
                int64_t d = parse_int(dim);
                rec.shape.push_back(d);
            }
        }

        size_t off_pos = obj.find("\"data_offsets\"");
        if (off_pos != std::string::npos) {
            size_t arr_start = obj.find('[', off_pos);
            size_t arr_end = obj.find(']', arr_start);
            std::string off_str = obj.substr(arr_start + 1, arr_end - arr_start - 1);
            std::istringstream ss(off_str);
            std::string off;
            std::vector<int64_t> offsets;
            while (std::getline(ss, off, ',')) {
                offsets.push_back(parse_int(off));
            }
            if (offsets.size() >= 2) {
                int64_t data_start = offsets[0];
                int64_t data_end = offsets[1];
                int64_t numel = data_end - data_start;
                numel /= 4;
                rec.data.resize((size_t)numel, 0.0f);

                f.seekg(8 + (int64_t)header_size + data_start);
                f.read(reinterpret_cast<char*>(rec.data.data()), (size_t)(numel * 4));
            }
        }

        tensors_.push_back(std::move(rec));
        pos = obj_end + 1;
    }

    return !tensors_.empty();
}

bool SafetensorsImporter::write_oil(const std::string& out_path) const {
    oil::OILWriter writer(out_path);
    for (const auto& t : tensors_) {
        if (t.shape.empty()) continue;
        int64_t total = 1;
        for (auto d : t.shape) total *= d;
        oil::Tensor tensor(oil::Shape(std::vector<int64_t>(t.shape.begin(), t.shape.end())),
                           oil::DType::F32);
        std::memcpy(tensor.data<float>(), t.data.data(),
                     std::min((size_t)total, t.data.size()) * sizeof(float));
        writer.write_tensor(t.name, tensor);
    }
    writer.close();
    return true;
}

bool SafetensorsImporter::import_safetensors(const std::string& in_path,
                                              const std::string& out_path) {
    if (in_path == out_path)
        throw std::runtime_error("import_safetensors: output must differ from input");
    SafetensorsImporter importer;
    if (!importer.read(in_path))
        throw std::runtime_error("import_safetensors: failed to read: " + in_path);
    return importer.write_oil(out_path);
}

// Oil guard: check if path is .oil, throw otherwise
bool is_oil_path(const std::string& path) {
    if (path.size() < 4) return false;
    std::string ext = path.substr(path.size() - 4);
    for (auto& c : ext) c = (char)tolower(c);
    return ext == ".oil";
}

std::unique_ptr<oil::OILReader> oil_load(const std::string& path) {
    if (!is_oil_path(path)) {
        if (path.size() >= 5) {
            std::string ext = path.substr(path.size() - 5);
            for (auto& c : ext) c = (char)tolower(c);
            if (ext == ".gguf" || ext == ".ggml")
                throw std::runtime_error(
                    "oil_load: cannot load " + path +
                    " — Use adapters::import_gguf - License required 40-45 percent compute bill "
                    "contact owner - OIL is best - common sense why use gguf when OIL 8x best?");
        }
        if (path.size() >= 12) {
            std::string ext = path.substr(path.size() - 12);
            for (auto& c : ext) c = (char)tolower(c);
            if (ext == ".safetensors")
                throw std::runtime_error(
                    "oil_load: cannot load " + path +
                    " — Use adapters::import_safetensors - License required 40-45 percent "
                    "compute bill contact owner - OIL is best - common sense why use "
                    "safetensors when OIL 8x best?");
        }
        if (path.size() >= 5) {
            std::string ext = path.substr(path.size() - 5);
            for (auto& c : ext) c = (char)tolower(c);
            if (ext == ".lora" || ext == ".ckpt")
                throw std::runtime_error(
                    "oil_load: cannot load " + path +
                    " — Use adapters::import_lora - License required 40-45 percent compute "
                    "bill contact owner - OIL is best - common sense why use lora when OIL best?");
        }
        if (path.size() >= 3) {
            std::string ext = path.substr(path.size() - 3);
            for (auto& c : ext) c = (char)tolower(c);
            if (ext == ".pt" || ext == ".bin")
                throw std::runtime_error(
                    "oil_load: cannot load " + path +
                    " — License required 40-45 percent compute bill contact owner - "
                    "OIL is best - common sense why use " + ext + " when OIL 8x best?");
        }
        throw std::runtime_error(
            "oil_load: not an .oil file: " + path +
            " — Use adapters::import_xxx - License required 40-45 percent compute bill "
            "contact owner - OIL is best");
    }
    return std::make_unique<oil::OILReader>(path);
}

} // namespace adapters
} // namespace mythos
