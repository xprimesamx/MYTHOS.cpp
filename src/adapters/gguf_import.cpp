#include "oil/adapters.h"

#include <cstring>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace mythos {
namespace adapters {

GGUFImporter::GGUFImporter() {}

bool GGUFImporter::read(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) return false;

    uint32_t version;
    f.read(reinterpret_cast<char*>(&version), 4);
    meta_.version = version;

    uint64_t tensor_count, kv_count;
    f.read(reinterpret_cast<char*>(&tensor_count), 8);
    f.read(reinterpret_cast<char*>(&kv_count), 8);
    meta_.tensor_count = tensor_count;
    meta_.kv_count = kv_count;

    for (uint64_t i = 0; i < kv_count && f.good(); ++i) {
        uint32_t key_len;
        f.read(reinterpret_cast<char*>(&key_len), 4);
        std::string key(key_len, '\0');
        f.read(&key[0], key_len);

        uint32_t val_type;
        f.read(reinterpret_cast<char*>(&val_type), 4);

        if (val_type == 8) {
            uint64_t str_len;
            f.read(reinterpret_cast<char*>(&str_len), 8);
            std::string val(str_len, '\0');
            f.read(&val[0], str_len);
            if (key == "architecture") meta_.arch = val;
        } else if (val_type == 4) {
            uint32_t val;
            f.read(reinterpret_cast<char*>(&val), 4);
        } else if (val_type == 5) {
            int32_t val;
            f.read(reinterpret_cast<char*>(&val), 4);
        } else if (val_type == 10) {
            uint64_t val;
            f.read(reinterpret_cast<char*>(&val), 8);
        } else if (val_type == 6) {
            float val;
            f.read(reinterpret_cast<char*>(&val), 4);
        } else if (val_type == 7) {
            bool val;
            f.read(reinterpret_cast<char*>(&val), 1);
        } else if (val_type == 2) {
            uint8_t val;
            f.read(reinterpret_cast<char*>(&val), 1);
        } else {
            break;
        }
    }

    for (uint64_t i = 0; i < tensor_count && f.good(); ++i) {
        uint32_t name_len;
        f.read(reinterpret_cast<char*>(&name_len), 4);
        std::string name(name_len, '\0');
        f.read(&name[0], name_len);

        uint32_t n_dims;
        f.read(reinterpret_cast<char*>(&n_dims), 4);

        TensorRecord rec;
        rec.name = name;
        rec.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t dim;
            f.read(reinterpret_cast<char*>(&dim), 8);
            rec.shape[d] = (int64_t)dim;
        }

        uint32_t dtype;
        f.read(reinterpret_cast<char*>(&dtype), 4);

        uint64_t offset;
        f.read(reinterpret_cast<char*>(&offset), 8);

        int64_t numel = 1;
        for (auto d : rec.shape) numel *= d;
        rec.data.resize((size_t)numel, 0.0f);
        tensors_.push_back(std::move(rec));
    }

    return f.good() || !tensors_.empty();
}

bool GGUFImporter::write_oil(const std::string& out_path) const {
    oil::OILWriter writer(out_path);
    for (const auto& t : tensors_) {
        int64_t total = 1;
        for (auto d : t.shape) total *= d;
        oil::Tensor tensor(oil::Shape{(int64_t)t.shape.size(), }, oil::DType::F32);
        if (!t.shape.empty()) {
            std::vector<int64_t> dims(t.shape.begin(), t.shape.end());
            tensor = oil::Tensor(oil::Shape(dims), oil::DType::F32);
            std::memcpy(tensor.data<float>(), t.data.data(), (size_t)total * sizeof(float));
        }
        writer.write_tensor(t.name, tensor);
    }
    writer.close();
    return true;
}

bool GGUFImporter::import_gguf(const std::string& in_path, const std::string& out_path) {
    if (in_path == out_path)
        throw std::runtime_error("import_gguf: output path must differ from input (never overwrite)");
    if (out_path.substr(out_path.size() - 4) != ".oil")
        throw std::runtime_error("import_gguf: output must be .oil format");
    if (in_path.substr(in_path.size() - 5) == ".gguf" && out_path == in_path)
        throw std::runtime_error("import_gguf: cannot overwrite source GGUF");

    GGUFImporter importer;
    if (!importer.read(in_path))
        throw std::runtime_error("import_gguf: failed to read GGUF file: " + in_path);
    return importer.write_oil(out_path);
}

} // namespace adapters
} // namespace mythos
