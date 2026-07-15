#include "oil/oil_format.h"
#include <cstring>
#include <vector>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace oil {

#if defined(_WIN32)
class MappedFile {
public:
    MappedFile() : hFile(INVALID_HANDLE_VALUE), hMap(nullptr), data(nullptr), size_(0) {}

    ~MappedFile() { close(); }

    bool open(const char* path) {
        hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) return false;

        LARGE_INTEGER li;
        GetFileSizeEx(hFile, &li);
        size_ = (size_t)li.QuadPart;

        hMap = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; return false; }

        data = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!data) { CloseHandle(hMap); hMap = NULL; CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; return false; }

        return true;
    }

    void close() {
        if (data) { UnmapViewOfFile(data); data = nullptr; }
        if (hMap) { CloseHandle(hMap); hMap = NULL; }
        if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
    }

    const uint8_t* ptr() const { return (const uint8_t*)data; }
    size_t size() const { return size_; }
    bool is_open() const { return data != nullptr; }

private:
    HANDLE hFile;
    HANDLE hMap;
    void* data;
    size_t size_;
};
#else
class MappedFile {
public:
    MappedFile() : data_(nullptr), size_(0) {}
    ~MappedFile() { delete[] data_; }

    bool open(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END);
        size_ = ftell(f);
        fseek(f, 0, SEEK_SET);
        data_ = new uint8_t[size_];
        fread(data_, 1, size_, f);
        fclose(f);
        return true;
    }

    const uint8_t* ptr() const { return data_; }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr; }

private:
    uint8_t* data_;
    size_t size_;
};
#endif

// ===========================================================================
// OILWriter
// ===========================================================================

OILWriter::OILWriter(const std::string& path)
    : file_(path, std::ios::binary), data_start_(0) {}

OILWriter::~OILWriter() {
    if (file_.is_open()) close();
}

void OILWriter::write_header(const OILHeader& hdr, const uint8_t* config_data) {
    if (!file_.is_open()) return;
    file_.write(hdr.magic, 4);
    file_.write((const char*)&hdr.version, sizeof(hdr.version));
    file_.write((const char*)&hdr.flags, sizeof(hdr.flags));
    file_.write((const char*)&hdr.config_size, sizeof(hdr.config_size));
    if (config_data && hdr.config_size > 0) {
        file_.write((const char*)config_data, hdr.config_size);
    }
    data_start_ = (size_t)file_.tellp();
}

void OILWriter::write_format_table(const std::vector<FormatBlockEntry>& entries) {
    if (!file_.is_open()) return;
    uint32_t num = (uint32_t)entries.size();
    file_.write((const char*)&num, sizeof(num));
    for (const auto& e : entries) {
        file_.write((const char*)&e.block_id, sizeof(e.block_id));
        file_.write((const char*)&e.format, sizeof(e.format));
        file_.write((const char*)&e.cb_bytes, sizeof(e.cb_bytes));
    }
}

void OILWriter::write_block(const BlockData& block) {
    if (!file_.is_open()) return;
    uint32_t nw = block.num_weights;
    file_.write((const char*)&nw, sizeof(nw));
    uint32_t cb_bytes = (uint32_t)block.codebook.size();
    file_.write((const char*)&cb_bytes, sizeof(cb_bytes));
    if (cb_bytes > 0) {
        file_.write((const char*)block.codebook.data(), cb_bytes);
    }
    uint32_t idx_bytes = (uint32_t)block.indices.size();
    file_.write((const char*)&idx_bytes, sizeof(idx_bytes));
    if (idx_bytes > 0) {
        file_.write((const char*)block.indices.data(), idx_bytes);
    }
}

void OILWriter::write_tensor_table(const std::vector<TensorEntry>& entries,
                                    const std::vector<std::string>& names) {
    if (!file_.is_open()) return;
    uint32_t num = (uint32_t)entries.size();
    file_.write((const char*)&num, sizeof(num));
    for (size_t i = 0; i < entries.size(); i++) {
        const auto& e = entries[i];
        uint16_t name_len = (uint16_t)(i < names.size() ? names[i].size() : 0);
        file_.write((const char*)&name_len, sizeof(name_len));
        if (name_len > 0) {
            file_.write(names[i].data(), name_len);
        }
        file_.write((const char*)&e.block_start, sizeof(e.block_start));
        file_.write((const char*)&e.num_blocks, sizeof(e.num_blocks));
    }
}

void OILWriter::close() {
    if (file_.is_open()) {
        file_.close();
    }
}

// ===========================================================================
// OILReader
// ===========================================================================

OILReader::OILReader(const std::string& path)
    : mapped_file_(nullptr), data_(nullptr), file_size_(0),
      format_table_offset_(0), tensor_table_offset_(0), data_offset_(0),
      num_format_blocks_(0), num_tensors_(0)
{
    MappedFile* mf = new MappedFile();
    if (!mf->open(path.c_str())) { delete mf; return; }
    mapped_file_ = mf;
    data_ = mapped_file_->ptr();
    file_size_ = mapped_file_->size();

    memcpy(&header_, data_, sizeof(OILHeader));

    if (memcmp(header_.magic, "OIL1", 4) != 0) return;

    format_table_offset_ = 16 + header_.config_size;

    const uint8_t* p = data_ + format_table_offset_;
    memcpy(&num_format_blocks_, p, sizeof(num_format_blocks_)); p += sizeof(uint32_t);

    size_t ft_bytes = sizeof(uint32_t) + (size_t)num_format_blocks_ * sizeof(FormatBlockEntry);
    tensor_table_offset_ = format_table_offset_ + ft_bytes;

    p = data_ + tensor_table_offset_;
    memcpy(&num_tensors_, p, sizeof(num_tensors_)); p += sizeof(uint32_t);

    const uint8_t* tt_start = p;
    for (uint32_t i = 0; i < num_tensors_; i++) {
        uint16_t name_len; memcpy(&name_len, tt_start, sizeof(name_len)); tt_start += sizeof(name_len);
        tt_start += name_len;
        tt_start += sizeof(uint32_t) + sizeof(uint32_t);
    }
    data_offset_ = (size_t)(tt_start - data_);

    p = data_ + format_table_offset_;
    memcpy(&num_format_blocks_, p, sizeof(num_format_blocks_)); p += sizeof(uint32_t);
    cached_ft_.resize(num_format_blocks_);
    for (uint32_t i = 0; i < num_format_blocks_; i++) {
        memcpy(&cached_ft_[i].block_id, p, sizeof(uint32_t)); p += sizeof(uint32_t);
        memcpy(&cached_ft_[i].format, p, sizeof(uint8_t)); p += sizeof(uint8_t);
        memcpy(&cached_ft_[i].cb_bytes, p, sizeof(uint32_t)); p += sizeof(uint32_t);
    }
}

OILReader::~OILReader() {
    delete mapped_file_;
}

const OILHeader& OILReader::header() const {
    return header_;
}

std::vector<uint8_t> OILReader::read_config() const {
    if (header_.config_size == 0 || !data_) return {};
    std::vector<uint8_t> cfg(header_.config_size);
    memcpy(cfg.data(), data_ + 16, header_.config_size);
    return cfg;
}

std::vector<FormatBlockEntry> OILReader::read_format_table() const {
    return cached_ft_;
}

size_t OILReader::num_blocks() const {
    return num_format_blocks_;
}

BlockData OILReader::read_block(uint32_t block_id) const {
    BlockData bd;
    bd.format = Format::TERNARY;

    size_t block_offset = data_offset_;
    for (uint32_t i = 0; i < block_id && i < num_format_blocks_; i++) {
        const uint8_t* p = data_ + block_offset;
        uint32_t nw; memcpy(&nw, p, sizeof(nw));
        uint32_t cb_bytes; memcpy(&cb_bytes, p + sizeof(nw), sizeof(cb_bytes));
        uint32_t idx_bytes; memcpy(&idx_bytes, p + sizeof(nw) + sizeof(cb_bytes) + cb_bytes, sizeof(idx_bytes));
        block_offset += sizeof(nw) + sizeof(cb_bytes) + cb_bytes + sizeof(idx_bytes) + idx_bytes;
    }

    const uint8_t* p = data_ + block_offset;
    uint32_t nw; memcpy(&nw, p, sizeof(nw)); p += sizeof(nw);
    bd.num_weights = nw;
    uint32_t cb_bytes; memcpy(&cb_bytes, p, sizeof(cb_bytes)); p += sizeof(cb_bytes);
    if (cb_bytes > 0) {
        bd.codebook.resize(cb_bytes);
        memcpy(bd.codebook.data(), p, cb_bytes); p += cb_bytes;
    }
    uint32_t idx_bytes; memcpy(&idx_bytes, p, sizeof(idx_bytes)); p += sizeof(idx_bytes);
    if (idx_bytes > 0) {
        bd.indices.resize(idx_bytes);
        memcpy(bd.indices.data(), p, idx_bytes);
    }

    if (block_id < (uint32_t)cached_ft_.size()) {
        switch (cached_ft_[block_id].format) {
            case 0: bd.format = Format::BINARY; break;
            case 1: bd.format = Format::TERNARY; break;
            case 2: bd.format = Format::OIL4; break;
            case 3: bd.format = Format::OIL8; break;
            default: bd.format = Format::FP32; break;
        }
    }

    return bd;
}

Tensor OILReader::read_tensor(const std::string& name) const {
    if (!data_) return Tensor();
    const uint8_t* p = data_ + tensor_table_offset_;
    uint32_t num_tensors; memcpy(&num_tensors, p, sizeof(num_tensors)); p += sizeof(uint32_t);

    for (uint32_t i = 0; i < num_tensors; i++) {
        uint16_t name_len; memcpy(&name_len, p, sizeof(name_len)); p += sizeof(name_len);
        std::string tensor_name((const char*)p, name_len); p += name_len;
        uint32_t block_start; memcpy(&block_start, p, sizeof(block_start)); p += sizeof(block_start);
        uint32_t num_blocks; memcpy(&num_blocks, p, sizeof(num_blocks)); p += sizeof(num_blocks);

        if (tensor_name == name) {
            int64_t total_weights = 0;
            for (uint32_t b = 0; b < num_blocks; b++) {
                BlockData bd = read_block(block_start + b);
                total_weights += bd.num_weights;
            }
            if (total_weights == 0) return Tensor();

            Tensor t(Shape{total_weights}, DType::F32);
            float* td = (float*)t.data();
            for (uint32_t b = 0; b < num_blocks; b++) {
                BlockData bd = read_block(block_start + b);
                if (bd.format == Format::FP32 && bd.indices.size() >= bd.num_weights * 4) {
                    memcpy(td, bd.indices.data(), bd.num_weights * 4);
                } else if (bd.format == Format::OIL8 && bd.codebook.size() >= 256 * 4) {
                    size_t tmp_off = 0;
                    CodebookOIL8 cb = CodebookOIL8::deserialize(bd.codebook.data(), tmp_off);
                    for (uint32_t j = 0; j < bd.num_weights; j++) {
                        td[j] = cb.dequantize(bd.indices[j]);
                    }
                } else if (bd.format == Format::OIL4 && bd.codebook.size() >= 16 * 2) {
                    size_t tmp_off = 0;
                    CodebookOIL4 cb = CodebookOIL4::deserialize(bd.codebook.data(), tmp_off);
                    for (uint32_t j = 0; j < bd.num_weights; j++) {
                        uint8_t idx;
                        if (j % 2 == 0) idx = bd.indices[j / 2] & 0x0F;
                        else idx = (bd.indices[j / 2] >> 4) & 0x0F;
                        td[j] = cb.dequantize(idx);
                    }
                } else {
                    for (uint32_t j = 0; j < bd.num_weights; j++) {
                        td[j] = 0;
                    }
                }
                td += bd.num_weights;
            }
            return t;
        }
    }
    return Tensor();
}

std::vector<std::string> OILReader::tensor_names() const {
    if (!data_) return {};
    std::vector<std::string> names;
    const uint8_t* p = data_ + tensor_table_offset_;
    uint32_t num_tensors; memcpy(&num_tensors, p, sizeof(num_tensors)); p += sizeof(uint32_t);
    for (uint32_t i = 0; i < num_tensors; i++) {
        uint16_t name_len; memcpy(&name_len, p, sizeof(name_len)); p += sizeof(name_len);
        std::string tensor_name((const char*)p, name_len); p += name_len;
        names.push_back(tensor_name);
        p += sizeof(uint32_t) + sizeof(uint32_t);
    }
    return names;
}

std::vector<Format> OILReader::tensor_formats(const std::string& name) const {
    if (!data_) return {};
    std::vector<Format> fmts;
    const uint8_t* p = data_ + tensor_table_offset_;
    uint32_t num_tensors; memcpy(&num_tensors, p, sizeof(num_tensors)); p += sizeof(uint32_t);
    for (uint32_t i = 0; i < num_tensors; i++) {
        uint16_t name_len; memcpy(&name_len, p, sizeof(name_len)); p += sizeof(name_len);
        std::string tensor_name((const char*)p, name_len); p += name_len;
        uint32_t block_start; memcpy(&block_start, p, sizeof(block_start)); p += sizeof(block_start);
        uint32_t num_blocks; memcpy(&num_blocks, p, sizeof(num_blocks)); p += sizeof(num_blocks);

        if (tensor_name == name) {
            for (uint32_t b = 0; b < num_blocks && (block_start + b) < (uint32_t)cached_ft_.size(); b++) {
                uint32_t bid = block_start + b;
                uint8_t fmt = cached_ft_[bid].format;
                switch (fmt) {
                    case 0: fmts.push_back(Format::BINARY); break;
                    case 1: fmts.push_back(Format::TERNARY); break;
                    case 2: fmts.push_back(Format::OIL4); break;
                    case 3: fmts.push_back(Format::OIL8); break;
                    case 4: fmts.push_back(Format::FP16); break;
                    case 5: fmts.push_back(Format::FP32); break;
                    default: fmts.push_back(Format::FP32); break;
                }
            }
            return fmts;
        }
    }
    return fmts;
}

} // namespace oil
