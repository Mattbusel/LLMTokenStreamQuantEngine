#pragma once
// VocabularyManager.h — Vocabulary management for token streams.

#include <cstddef>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

// ---------------------------------------------------------------------------
// VocabEntry: one entry in the vocabulary
// ---------------------------------------------------------------------------

struct VocabEntry {
    std::string token;
    std::size_t id{0};
    std::size_t frequency{0};
    bool is_special{false};
};

// ---------------------------------------------------------------------------
// VocabularyManager
// ---------------------------------------------------------------------------

/// Manages a vocabulary: maps tokens to integer IDs and back.
/// Unknown tokens are encoded as the UNK id (added automatically).
class VocabularyManager {
public:
    static constexpr const char* UNK_TOKEN = "<UNK>";

    explicit VocabularyManager();
    ~VocabularyManager() = default;

    VocabularyManager(const VocabularyManager&) = delete;
    VocabularyManager& operator=(const VocabularyManager&) = delete;
    VocabularyManager(VocabularyManager&&) = default;
    VocabularyManager& operator=(VocabularyManager&&) = default;

    /// Add a token to the vocabulary. Returns its assigned ID.
    /// If the token already exists, increments its frequency and returns its ID.
    std::size_t add_token(const std::string& token, bool is_special = false);

    /// Look up the ID for a token.  Returns std::nullopt if not found.
    std::optional<std::size_t> token_to_id(const std::string& token) const;

    /// Look up the token string for an ID.  Returns std::nullopt if not found.
    std::optional<std::string> id_to_token(std::size_t id) const;

    /// Encode a sequence of tokens to IDs.  Unknown tokens map to the UNK ID.
    std::vector<std::size_t> encode(const std::vector<std::string>& tokens) const;

    /// Decode a sequence of IDs to token strings.  Unknown IDs map to UNK_TOKEN.
    std::vector<std::string> decode(const std::vector<std::size_t>& ids) const;

    /// Return current vocabulary size (including special tokens).
    std::size_t size() const noexcept { return entries_.size(); }

    /// Remove tokens with frequency below min_frequency (special tokens are retained).
    void prune(std::size_t min_frequency);

    /// Write vocabulary to a TSV file: "id\ttoken\tfreq\n" per line.
    void save_tsv(const std::string& path) const;

    /// Return the UNK token's ID.
    std::size_t unk_id() const noexcept { return unk_id_; }

    /// Return all entries sorted by id ascending.
    std::vector<VocabEntry> entries_sorted() const;

private:
    std::vector<VocabEntry> entries_;                  // indexed by id
    std::unordered_map<std::string, std::size_t> token_to_id_; // token -> index into entries_
    std::size_t unk_id_{0};
};

} // namespace llmquant
