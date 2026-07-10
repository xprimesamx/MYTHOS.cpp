#include "oil/tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>

namespace oil {

BPETokenizer::BPETokenizer() {
    for (int i = 0; i < 256; i++) {
        vocab_.push_back(std::string(1, (char)i));
    }
    bos_id_ = 1;
    eos_id_ = 2;
    unk_id_ = 0;
}

void BPETokenizer::add_token(const std::string& token) {
    vocab_.push_back(token);
}

int BPETokenizer::get_token_id(const std::string& token) const {
    for (size_t i = 0; i < vocab_.size(); i++) {
        if (vocab_[i] == token) return (int)i;
    }
    return unk_id_;
}

std::vector<int> BPETokenizer::bpe_merge(const std::vector<int>& ids) const {
    std::vector<int> result = ids;
    while (true) {
        int best_left = -1, best_right = -1, best_id = -1;
        for (size_t i = 0; i + 1 < result.size(); i++) {
            auto it = merges_.find({result[i], result[i+1]});
            if (it != merges_.end()) {
                best_left = result[i];
                best_right = result[i+1];
                best_id = it->second;
                break;
            }
        }
        if (best_id == -1) break;
        std::vector<int> next;
        for (size_t i = 0; i < result.size(); i++) {
            if (i + 1 < result.size() && result[i] == best_left && result[i+1] == best_right) {
                next.push_back(best_id);
                i++;
            } else {
                next.push_back(result[i]);
            }
        }
        result = next;
    }
    return result;
}

void BPETokenizer::train(const std::vector<std::string>& texts, int vocab_size) {
    std::vector<int> all_bytes;
    for (const auto& t : texts) {
        for (char c : t) all_bytes.push_back((unsigned char)c);
    }

    int current_size = 256;
    while (current_size < vocab_size) {
        std::map<std::pair<int,int>, int> pair_counts;
        for (size_t i = 0; i + 1 < all_bytes.size(); i++) {
            pair_counts[{all_bytes[i], all_bytes[i+1]}]++;
        }
        if (pair_counts.empty()) break;

        auto best = pair_counts.begin();
        for (auto it = pair_counts.begin(); it != pair_counts.end(); ++it) {
            if (it->second > best->second) best = it;
        }

        int new_id = current_size++;
        merges_[best->first] = new_id;
        add_token(vocab_[best->first.first] + vocab_[best->first.second]);

        std::vector<int> merged;
        for (size_t i = 0; i < all_bytes.size(); i++) {
            if (i + 1 < all_bytes.size() &&
                all_bytes[i] == best->first.first &&
                all_bytes[i+1] == best->first.second) {
                merged.push_back(new_id);
                i++;
            } else {
                merged.push_back(all_bytes[i]);
            }
        }
        all_bytes = merged;
    }
}

std::vector<int> BPETokenizer::encode(const std::string& text) {
    std::vector<int> ids;
    for (char c : text) ids.push_back((unsigned char)c);
    return bpe_merge(ids);
}

std::string BPETokenizer::decode(const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) {
        if (id >= 0 && id < (int)vocab_.size()) {
            result += vocab_[id];
        }
    }
    return result;
}

void BPETokenizer::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    int vs = (int)vocab_.size();
    f.write((char*)&vs, sizeof(vs));
    for (const auto& v : vocab_) {
        int len = (int)v.size();
        f.write((char*)&len, sizeof(len));
        f.write(v.data(), len);
    }
    int ms = (int)merges_.size();
    f.write((char*)&ms, sizeof(ms));
    for (const auto& m : merges_) {
        f.write((char*)&m.first.first, sizeof(int));
        f.write((char*)&m.first.second, sizeof(int));
        f.write((char*)&m.second, sizeof(int));
    }
}

void BPETokenizer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    int vs; f.read((char*)&vs, sizeof(vs));
    vocab_.resize(vs);
    for (int i = 0; i < vs; i++) {
        int len; f.read((char*)&len, sizeof(len));
        vocab_[i].resize(len);
        f.read(&vocab_[i][0], len);
    }
    int ms; f.read((char*)&ms, sizeof(ms));
    merges_.clear();
    for (int i = 0; i < ms; i++) {
        int a, b, c; f.read((char*)&a, sizeof(a));
        f.read((char*)&b, sizeof(b));
        f.read((char*)&c, sizeof(c));
        merges_[{a,b}] = c;
    }
}

} // namespace oil
