#include "oil/oil_format.h"
#include <cstring>
#include <vector>
#include <string>

namespace oil {

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
    : file_(path, std::ios::binary), format_table_offset_(0), tensor_table_offset_(0),
      data_offset_(0), num_format_blocks_(0), num_tensors_(0) {
    if (!file_.is_open()) return;

    file_.read(header_.magic, 4);
    file_.read((char*)&header_.version, sizeof(header_.version));
    file_.read((char*)&header_.flags, sizeof(header_.flags));
    file_.read((char*)&header_.config_size, sizeof(header_.config_size));

    if (memcmp(header_.magic, "OIL1", 4) != 0) {
        return;
    }

    format_table_offset_ = 16 + header_.config_size;
    file_.seekg(format_table_offset_);
    file_.read((char*)&num_format_blocks_, sizeof(num_format_blocks_));

    size_t ft_bytes = sizeof(uint32_t) + num_format_blocks_ * sizeof(FormatBlockEntry);
    tensor_table_offset_ = format_table_offset_ + ft_bytes;
    file_.seekg(tensor_table_offset_);
    file_.read((char*)&num_tensors_, sizeof(num_tensors_));

    size_t tt_start = tensor_table_offset_ + sizeof(uint32_t);
    size_t tt_size = sizeof(uint32_t);
    file_.seekg(tt_start);
    for (uint32_t i = 0; i < num_tensors_; i++) {
        uint16_t name_len;
        file_.read((char*)&name_len, sizeof(name_len));
        tt_size += sizeof(name_len) + name_len + sizeof(TensorEntry) - sizeof(uint16_t);
        file_.seekg(tt_start + (std::streamoff)tt_size - (std::streamoff)(sizeof(uint16_t) + name_len + sizeof(TensorEntry) - sizeof(uint16_t)) + (std::streamoff)sizeof(uint16_t) + name_len + sizeof(TensorEntry) - sizeof(uint16_t));
        // simpler: just skip entries
    }
    // Re-calculate properly
    file_.seekg(tt_start);
    for (uint32_t i = 0; i < num_tensors_; i++) {
        uint16_t name_len;
        file_.read((char*)&name_len, sizeof(name_len));
        file_.seekg(name_len, std::ios::cur);
        file_.seekg(sizeof(TensorEntry) - sizeof(uint16_t), std::ios::cur);
    }
    data_offset_ = (size_t)file_.tellg();

    file_.clear();
    file_.seekg(format_table_offset_);
    cached_ft_.clear();
    file_.read((char*)&num_format_blocks_, sizeof(num_format_blocks_));
    cached_ft_.resize(num_format_blocks_);
    for (uint32_t i = 0; i < num_format_blocks_; i++) {
        FormatBlockEntry e;
        file_.read((char*)&e.block_id, sizeof(e.block_id));
        file_.read((char*)&e.format, sizeof(e.format));
        file_.read((char*)&e.cb_bytes, sizeof(e.cb_bytes));
        cached_ft_[i] = e;
    }
}

OILReader::~OILReader() {
    if (file_.is_open()) file_.close();
}

const OILHeader& OILReader::header() const {
    return header_;
}

std::vector<uint8_t> OILReader::read_config() const {
    if (header_.config_size == 0) return {};
    std::vector<uint8_t> cfg(header_.config_size);
    file_.seekg(16);
    file_.read((char*)cfg.data(), header_.config_size);
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
        file_.seekg((std::streamoff)block_offset);
        uint32_t nw; file_.read((char*)&nw, sizeof(nw));
        bd.num_weights = nw;
        uint32_t cb_bytes; file_.read((char*)&cb_bytes, sizeof(cb_bytes));
        block_offset += sizeof(nw) + sizeof(cb_bytes) + cb_bytes;
        file_.seekg(cb_bytes, std::ios::cur);
        uint32_t idx_bytes; file_.read((char*)&idx_bytes, sizeof(idx_bytes));
        block_offset += sizeof(idx_bytes) + idx_bytes;
    }

    file_.seekg((std::streamoff)block_offset);
    uint32_t nw; file_.read((char*)&nw, sizeof(nw));
    bd.num_weights = nw;
    uint32_t cb_bytes; file_.read((char*)&cb_bytes, sizeof(cb_bytes));
    if (cb_bytes > 0) {
        bd.codebook.resize(cb_bytes);
        file_.read((char*)bd.codebook.data(), cb_bytes);
    }
    uint32_t idx_bytes; file_.read((char*)&idx_bytes, sizeof(idx_bytes));
    if (idx_bytes > 0) {
        bd.indices.resize(idx_bytes);
        file_.read((char*)bd.indices.data(), idx_bytes);
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
    file_.seekg((std::streamoff)tensor_table_offset_ + sizeof(uint32_t));
    uint32_t num_tensors;
    file_.seekg((std::streamoff)tensor_table_offset_);
    file_.read((char*)&num_tensors, sizeof(num_tensors));

    for (uint32_t i = 0; i < num_tensors; i++) {
        uint16_t name_len;
        file_.read((char*)&name_len, sizeof(name_len));
        std::string tensor_name((size_t)name_len, '\0');
        if (name_len > 0) file_.read(&tensor_name[0], name_len);
        TensorEntry te;
        file_.read((char*)&te.block_start, sizeof(te.block_start));
        file_.read((char*)&te.num_blocks, sizeof(te.num_blocks));

        if (tensor_name == name) {
            int64_t total_weights = 0;
            for (uint32_t b = 0; b < te.num_blocks; b++) {
                BlockData bd = read_block(te.block_start + b);
                total_weights += bd.num_weights;
            }
            if (total_weights == 0) return Tensor();

            Tensor t(Shape{total_weights}, DType::F32);
            float* td = (float*)t.data();
            for (uint32_t b = 0; b < te.num_blocks; b++) {
                BlockData bd = read_block(te.block_start + b);
                if (bd.format == Format::FP32 && bd.indices.size() >= bd.num_weights * 4) {
                    memcpy(td, bd.indices.data(), bd.num_weights * 4);
                } else if (bd.format == Format::OIL8 && bd.codebook.size() >= 256 * 4) {
                    CodebookOIL8 cb = CodebookOIL8::deserialize(bd.codebook.data(), *(size_t*)0);
                    size_t tmp_off = 0;
                    cb = CodebookOIL8::deserialize(bd.codebook.data(), tmp_off);
                    for (uint32_t j = 0; j < bd.num_weights; j++) {
                        td[j] = cb.dequantize(bd.indices[j]);
                    }
                } else if (bd.format == Format::OIL4 && bd.codebook.size() >= 16 * 2) {
                    CodebookOIL4 cb = CodebookOIL4::deserialize(bd.codebook.data(), *(size_t*)0);
                    size_t tmp_off = 0;
                    cb = CodebookOIL4::deserialize(bd.codebook.data(), tmp_off);
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
    std::vector<std::string> names;
    file_.seekg((std::streamoff)tensor_table_offset_);
    uint32_t num_tensors;
    file_.read((char*)&num_tensors, sizeof(num_tensors));
    for (uint32_t i = 0; i < num_tensors; i++) {
        uint16_t name_len;
        file_.read((char*)&name_len, sizeof(name_len));
        std::string tensor_name((size_t)name_len, '\0');
        if (name_len > 0) file_.read(&tensor_name[0], name_len);
        names.push_back(tensor_name);
        file_.seekg(sizeof(TensorEntry) - sizeof(uint16_t), std::ios::cur);
    }
    return names;
}

std::vector<Format> OILReader::tensor_formats(const std::string& name) const {
    std::vector<Format> fmts;
    file_.seekg((std::streamoff)tensor_table_offset_);
    uint32_t num_tensors;
    file_.read((char*)&num_tensors, sizeof(num_tensors));
    for (uint32_t i = 0; i < num_tensors; i++) {
        uint16_t name_len;
        file_.read((char*)&name_len, sizeof(name_len));
        std::string tensor_name((size_t)name_len, '\0');
        if (name_len > 0) file_.read(&tensor_name[0], name_len);
        TensorEntry te;
        file_.read((char*)&te.block_start, sizeof(te.block_start));
        file_.read((char*)&te.num_blocks, sizeof(te.num_blocks));

        if (tensor_name == name) {
            for (uint32_t b = 0; b < te.num_blocks && (te.block_start + b) < (uint32_t)cached_ft_.size(); b++) {
                uint32_t bid = te.block_start + b;
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
