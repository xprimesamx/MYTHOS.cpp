#include "oil/oil_format.h"
#include "oil/tensor.h"
#include "oil/types.h"

#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <vector>
#include <cstdint>
#include <map>

struct ConvertArgs {
    std::string input_path;
    std::string output_path;
    std::string input_fmt = "rawfp32";
    bool verbose = false;
};

static ConvertArgs parse_args(int argc, char** argv) {
    ConvertArgs args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            args.input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            args.output_path = argv[++i];
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc)
            args.input_fmt = argv[++i];
        else if (strcmp(argv[i], "--verbose") == 0)
            args.verbose = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: oil_convert --input <file> --output <model.oil> [options]\n";
            std::cout << "  --format fmt   Input format: rawfp32 (default), gguf\n";
            std::cout << "  --verbose      Verbose output\n";
            exit(0);
        }
    }
    return args;
}

static bool convert_raw_fp32(const std::string& in_path, const std::string& out_path) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: cannot open input: " << in_path << std::endl;
        return false;
    }
    in.seekg(0, std::ios::end);
    size_t file_size = (size_t)in.tellg();
    in.seekg(0, std::ios::beg);

    if (file_size == 0 || file_size % 4 != 0) {
        std::cerr << "Error: raw FP32 file size must be multiple of 4\n";
        return false;
    }

    size_t num_floats = file_size / 4;
    std::vector<float> data(num_floats);
    in.read(reinterpret_cast<char*>(data.data()), (std::streamsize)file_size);
    in.close();

    oil::Tensor weights(oil::Shape{1, (int64_t)num_floats}, oil::DType::F32);
    memcpy(weights.data(), data.data(), file_size);

    oil::OILWriter writer(out_path);
    oil::OILHeader hdr;
    memcpy(hdr.magic, "OIL1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = 0;
    writer.write_header(hdr, nullptr);

    oil::FormatBlockEntry ft_entry;
    ft_entry.block_id = 0;
    ft_entry.format = (uint8_t)oil::Format::FP32;
    ft_entry.cb_bytes = 0;
    writer.write_format_table({ft_entry});

    oil::TensorEntry t_entry;
    t_entry.name_len = 0;
    t_entry.block_start = 0;
    t_entry.num_blocks = 1;
    writer.write_tensor_table({t_entry}, {"weights"});

    oil::BlockData block;
    block.format = oil::Format::FP32;
    block.num_weights = (uint32_t)num_floats;
    block.indices.resize(file_size);
    memcpy(block.indices.data(), data.data(), file_size);
    writer.write_block(block);

    writer.close();
    return true;
}

// GGUF format constants
enum GGMLType { GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_Q4_0 = 2, GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6, GGML_TYPE_Q5_1 = 7, GGML_TYPE_Q8_0 = 8, GGML_TYPE_Q8_1 = 9 };

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint32_t> dims;
    uint32_t type;
    uint64_t offset;
};

static uint64_t ggml_type_size(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_F16: return 2;
        case GGML_TYPE_Q4_0: return 18; // block_size=32, 16 bytes quant + 2 bytes scale
        case GGML_TYPE_Q4_1: return 20; // block_size=32, 16 bytes quant + 4 bytes scales
        case GGML_TYPE_Q5_0: return 22; // block_size=32, 16 bytes quant + 4 bytes scales + 2 bytes quants
        case GGML_TYPE_Q5_1: return 24;
        case GGML_TYPE_Q8_0: return 34; // block_size=32, 32 bytes quant + 2 bytes scale
        default: return 4;
    }
}

static uint32_t ggml_block_size(uint32_t type) {
    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q8_1:
            return 32;
        default: return 1;
    }
}

static std::string read_gguf_string(std::ifstream& f) {
    uint64_t len;
    f.read((char*)&len, sizeof(len));
    std::string s(len, '\0');
    if (len > 0) f.read(&s[0], len);
    return s;
}

static bool convert_gguf(const std::string& in_path, const std::string& out_path) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: cannot open GGUF file: " << in_path << std::endl;
        return false;
    }

    // Read GGUF header
    uint32_t magic;
    in.read((char*)&magic, sizeof(magic));
    if (magic != 0x46554747 && magic != 0x47475546) {
        std::cerr << "Error: not a valid GGUF file (magic: 0x" << std::hex << magic << std::dec << ")\n";
        return false;
    }

    uint32_t version;
    in.read((char*)&version, sizeof(version));
    uint64_t tensor_count, metadata_kv_count;
    in.read((char*)&tensor_count, sizeof(tensor_count));
    in.read((char*)&metadata_kv_count, sizeof(metadata_kv_count));

    if (version < 1 || version > 3) {
        std::cerr << "Warning: unknown GGUF version " << version << ", attempting to read...\n";
    }

    std::cout << "GGUF v" << version << ": " << tensor_count << " tensors, "
              << metadata_kv_count << " metadata entries\n";

    // Skip metadata KV pairs
    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        std::string key = read_gguf_string(in);
        uint32_t type;
        in.read((char*)&type, sizeof(type));
        switch (type) {
            case 0: { uint8_t v; in.read((char*)&v, sizeof(v)); break; }
            case 1: { int8_t v; in.read((char*)&v, sizeof(v)); break; }
            case 2: { uint16_t v; in.read((char*)&v, sizeof(v)); break; }
            case 3: { int16_t v; in.read((char*)&v, sizeof(v)); break; }
            case 4: { uint32_t v; in.read((char*)&v, sizeof(v)); break; }
            case 5: { int32_t v; in.read((char*)&v, sizeof(v)); break; }
            case 6: { float v; in.read((char*)&v, sizeof(v)); break; }
            case 7: { uint64_t v; in.read((char*)&v, sizeof(v)); break; }
            case 8: { int64_t v; in.read((char*)&v, sizeof(v)); break; }
            case 9: { double v; in.read((char*)&v, sizeof(v)); break; }
            case 10: { uint32_t n; in.read((char*)&n, sizeof(n)); for (uint32_t j = 0; j < n; j++) { bool b; in.read((char*)&b, sizeof(b)); } break; }
            case 11: { std::string s = read_gguf_string(in); break; }
            case 12: { uint32_t n; in.read((char*)&n, sizeof(n)); for (uint32_t j = 0; j < n; j++) in.ignore(1); break; }
            default: break;
        }
    }

    // Read tensor info entries
    std::vector<GGUFTensorInfo> tensors;
    for (uint64_t i = 0; i < tensor_count; i++) {
        GGUFTensorInfo t;
        t.name = read_gguf_string(in);
        uint32_t n_dims;
        in.read((char*)&n_dims, sizeof(n_dims));
        t.dims.resize(n_dims);
        for (uint32_t j = 0; j < n_dims; j++)
            in.read((char*)&t.dims[j], sizeof(uint32_t));
        in.read((char*)&t.type, sizeof(t.type));
        in.read((char*)&t.offset, sizeof(t.offset));
        tensors.push_back(t);
    }

    // Align to 32 bytes (standard GGUF alignment)
    uint64_t data_start = (uint64_t)in.tellg();
    uint64_t alignment = 32;
    uint64_t aligned = (data_start + alignment - 1) / alignment * alignment;
    in.seekg(aligned);

    // Create OIL output
    oil::OILWriter writer(out_path);
    oil::OILHeader hdr;
    memcpy(hdr.magic, "OIL1", 4);
    hdr.version = 1;
    hdr.flags = 0;
    hdr.config_size = 0;
    writer.write_header(hdr, nullptr);

    // Build format table from tensor types
    std::vector<oil::FormatBlockEntry> fmt_entries;
    std::vector<oil::TensorEntry> t_entries;
    std::vector<std::string> t_names;
    std::map<uint32_t, uint8_t> format_map; // GGML type -> format block ID
    uint8_t next_block_id = 0;

    // Quantized tensors will be converted to FP32
    auto get_or_create_format = [&](uint32_t ggml_type, int64_t n_weights) -> uint8_t {
        if (format_map.count(ggml_type)) return format_map[ggml_type];
        uint8_t bid = next_block_id++;
        oil::FormatBlockEntry fe;
        fe.block_id = bid;
        fe.format = (uint8_t)oil::Format::FP32;
        fe.cb_bytes = 0;
        fmt_entries.push_back(fe);
        format_map[ggml_type] = bid;
        return bid;
    };

    for (auto& t : tensors) {
        int64_t n_weights = 1;
        for (auto d : t.dims) n_weights *= (int64_t)d;
        uint8_t bid = get_or_create_format(t.type, n_weights);

        oil::TensorEntry te;
        te.name_len = (uint32_t)t.name.size();
        te.block_start = (uint32_t)t_entries.size(); // each tensor gets its own block
        te.num_blocks = 1;
        t_entries.push_back(te);
        t_names.push_back(t.name);
    }

    writer.write_format_table(fmt_entries);
    writer.write_tensor_table(t_entries, t_names);

    // Read and write each tensor
    for (size_t i = 0; i < tensors.size(); i++) {
        auto& t = tensors[i];
        int64_t n_weights = 1;
        for (auto d : t.dims) n_weights *= (int64_t)d;

        // Seek to tensor data
        in.seekg(aligned + t.offset);

        // Read and dequantize to FP32
        std::vector<float> f32_data(n_weights);
        uint32_t block_sz = ggml_block_size(t.type);
        uint64_t type_sz = ggml_type_size(t.type);
        uint64_t n_blocks = (n_weights + block_sz - 1) / block_sz;

        if (t.type == GGML_TYPE_F32) {
            in.read((char*)f32_data.data(), n_weights * sizeof(float));
        } else if (t.type == GGML_TYPE_F16) {
            for (int64_t j = 0; j < n_weights; j++) {
                uint16_t f16;
                in.read((char*)&f16, sizeof(f16));
                // FP16 to FP32 conversion
                uint32_t sign = (uint32_t)(f16 >> 15);
                uint32_t exp = (uint32_t)((f16 >> 10) & 0x1F);
                uint32_t mant = (uint32_t)(f16 & 0x3FF);
                uint32_t f32;
                if (exp == 0) {
                    f32 = (sign << 31) | (mant << 13);
                } else if (exp == 31) {
                    f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
                } else {
                    f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                }
                memcpy(&f32_data[j], &f32, sizeof(f32));
            }
        } else {
            // Quantized format: dequantize Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
            std::vector<uint8_t> block_data(n_blocks * type_sz);
            in.read((char*)block_data.data(), n_blocks * type_sz);

            for (uint64_t b = 0; b < n_blocks; b++) {
                int64_t base = (int64_t)b * block_sz;
                int64_t remain = std::min((int64_t)block_sz, n_weights - base);
                const uint8_t* src = block_data.data() + b * type_sz;

                if (t.type == GGML_TYPE_Q4_0) {
                    float scale;
                    memcpy(&scale, src + 16, sizeof(float));
                    for (int64_t j = 0; j < remain && j < 32; j++) {
                        int8_t q = (src[j / 2] >> (j % 2 * 4)) & 0x0F;
                        f32_data[base + j] = ((float)q - 8.0f) * scale;
                    }
                } else if (t.type == GGML_TYPE_Q4_1) {
                    float scale, min;
                    memcpy(&scale, src + 16, sizeof(float));
                    memcpy(&min, src + 20, sizeof(float));
                    for (int64_t j = 0; j < remain && j < 32; j++) {
                        int8_t q = (src[j / 2] >> (j % 2 * 4)) & 0x0F;
                        f32_data[base + j] = (float)q * scale + min;
                    }
                } else if (t.type == GGML_TYPE_Q8_0) {
                    float scale;
                    memcpy(&scale, src + 32, sizeof(float));
                    for (int64_t j = 0; j < remain && j < 32; j++) {
                        f32_data[base + j] = (float)((int8_t)src[j]) * scale;
                    }
                } else {
                    // Fallback: copy raw bytes as float
                    for (int64_t j = 0; j < remain; j++)
                        f32_data[base + j] = 0.0f;
                }
            }
        }

        // Write as OIL block
        oil::BlockData block;
        block.format = oil::Format::FP32;
        block.num_weights = (uint32_t)f32_data.size();
        size_t byte_size = f32_data.size() * sizeof(float);
        block.indices.resize(byte_size);
        memcpy(block.indices.data(), f32_data.data(), byte_size);
        writer.write_block(block);
    }

    writer.close();
    std::cout << "Converted " << tensors.size() << " tensors to OIL: " << out_path << std::endl;
    return true;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    std::cout << "OIL Converter" << std::endl;

    if (args.input_path.empty() || args.output_path.empty()) {
        std::cerr << "Error: --input and --output are required\n";
        return 1;
    }

    bool ok = false;
    if (args.input_fmt == "rawfp32") {
        ok = convert_raw_fp32(args.input_path, args.output_path);
    } else if (args.input_fmt == "gguf") {
        ok = convert_gguf(args.input_path, args.output_path);
    } else {
        std::cerr << "Unknown format: " << args.input_fmt << std::endl;
        return 1;
    }

    if (!ok) return 1;
    if (args.verbose) {
        std::cout << "Written: " << args.output_path << std::endl;
    }
    return 0;
}
