#pragma once
#include "oil/types.h"
#include "oil/tensor.h"
#include "oil/codebook.h"
#include <string>
#include <vector>
#include <fstream>

namespace oil {

#pragma pack(push, 1)
struct OILHeader {
    char magic[4];      // "OIL1"
    uint32_t version;   // packed: major<<22 | minor<<12 | patch
    uint32_t flags;
    uint32_t config_size;
};

struct FormatBlockEntry {
    uint32_t block_id;
    uint8_t format;    // 0=binary, 1=ternary, 2=oil4, 3=oil8
    uint32_t cb_bytes; // codebook size in bytes
};

struct TensorEntry {
    uint16_t name_len;
    uint32_t block_start;
    uint32_t num_blocks;
};
#pragma pack(pop)

struct BlockData {
    Format format;
    std::vector<uint8_t> codebook;
    std::vector<uint8_t> indices;
    uint32_t num_weights;
};

class OILWriter {
public:
    explicit OILWriter(const std::string& path);
    ~OILWriter();
    
    void write_header(const OILHeader& hdr, const uint8_t* config_data);
    void write_format_table(const std::vector<FormatBlockEntry>& entries);
    void write_block(const BlockData& block);
    void write_tensor_table(const std::vector<TensorEntry>& entries,
                            const std::vector<std::string>& names);
    void close();
    
private:
    std::ofstream file_;
    size_t data_start_;
};

class OILReader {
public:
    explicit OILReader(const std::string& path);
    ~OILReader();
    
    const OILHeader& header() const;
    std::vector<uint8_t> read_config() const;
    std::vector<FormatBlockEntry> read_format_table() const;
    BlockData read_block(uint32_t block_id) const;
    size_t num_blocks() const;
    
    // Read a named tensor's blocks and reconstruct
    Tensor read_tensor(const std::string& name) const;
    std::vector<std::string> tensor_names() const;
    
    // Get format info for a tensor
    std::vector<Format> tensor_formats(const std::string& name) const;
    
private:
    OILHeader header_;
    size_t format_table_offset_;
    size_t tensor_table_offset_;
    size_t data_offset_;
    uint32_t num_format_blocks_;
    uint32_t num_tensors_;
    mutable std::ifstream file_;
    mutable std::vector<FormatBlockEntry> cached_ft_;
};

} // namespace oil
