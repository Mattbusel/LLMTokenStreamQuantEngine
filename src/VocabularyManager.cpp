// VocabularyManager.cpp — Implementation of VocabularyManager.

#include "VocabularyManager.h"

#include <algorithm>
#include <stdexcept>

namespace llmquant {

VocabularyManager::VocabularyManager() {
    // Always add UNK as the first token (id = 0).
    VocabEntry unk_entry;
    unk_entry.token = UNK_TOKEN;
    unk_entry.id = 0;
    unk_entry.frequency = 0;
    unk_entry.is_special = true;
    entries_.push_back(std::move(unk_entry));
    token_to_id_[UNK_TOKEN] = 0;
    unk_id_ = 0;
}

std::size_t VocabularyManager::add_token(const std::string& token, bool is_special) {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        // Token exists — just bump frequency
        entries_[it->second].frequency += 1;
        return it->second;
    }
    // New token
    std::size_t new_id = entries_.size();
    VocabEntry entry;
    entry.token = token;
    entry.id = new_id;
    entry.frequency = 1;
    entry.is_special = is_special;
    entries_.push_back(std::move(entry));
    token_to_id_[token] = new_id;
    return new_id;
}

std::optional<std::size_t> VocabularyManager::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it == token_to_id_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<std::string> VocabularyManager::id_to_token(std::size_t id) const {
    if (id >= entries_.size()) {
        return std::nullopt;
    }
    return entries_[id].token;
}

std::vector<std::size_t> VocabularyManager::encode(
    const std::vector<std::string>& tokens) const {
    std::vector<std::size_t> ids;
    ids.reserve(tokens.size());
    for (const auto& tok : tokens) {
        auto it = token_to_id_.find(tok);
        ids.push_back(it != token_to_id_.end() ? it->second : unk_id_);
    }
    return ids;
}

std::vector<std::string> VocabularyManager::decode(
    const std::vector<std::size_t>& ids) const {
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (std::size_t id : ids) {
        if (id < entries_.size()) {
            tokens.push_back(entries_[id].token);
        } else {
            tokens.push_back(UNK_TOKEN);
        }
    }
    return tokens;
}

void VocabularyManager::prune(std::size_t min_frequency) {
    // Mark entries for removal (special tokens are always kept)
    // We need to rebuild entries_ and token_to_id_ with new contiguous IDs.
    std::vector<VocabEntry> kept;
    std::unordered_map<std::string, std::size_t> new_map;

    // UNK always kept as id=0
    std::size_t new_id = 0;
    for (auto& entry : entries_) {
        if (entry.is_special || entry.frequency >= min_frequency) {
            entry.id = new_id;
            new_map[entry.token] = new_id;
            kept.push_back(entry);
            ++new_id;
        }
    }
    entries_ = std::move(kept);
    token_to_id_ = std::move(new_map);

    // Ensure UNK id is still correct
    auto unk_it = token_to_id_.find(UNK_TOKEN);
    unk_id_ = (unk_it != token_to_id_.end()) ? unk_it->second : 0;
}

void VocabularyManager::save_tsv(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("VocabularyManager::save_tsv: cannot open " + path);
    }
    for (const auto& entry : entries_) {
        ofs << entry.id << '\t' << entry.token << '\t' << entry.frequency << '\n';
    }
}

std::vector<VocabEntry> VocabularyManager::entries_sorted() const {
    std::vector<VocabEntry> sorted = entries_;
    std::sort(sorted.begin(), sorted.end(),
              [](const VocabEntry& a, const VocabEntry& b) { return a.id < b.id; });
    return sorted;
}

} // namespace llmquant
