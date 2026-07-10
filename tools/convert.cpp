#include "oil/oil_format.h"
#include "oil/tensor.h"
#include "oil/types.h"

#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <vector>
#include <cstdint>

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

static bool convert_gguf(const std::string& in_path, const std::string& out_path) {
    std::cerr << "GGUF conversion not yet implemented; use --format rawfp32\n";
    (void)out_path;
    return false;
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
