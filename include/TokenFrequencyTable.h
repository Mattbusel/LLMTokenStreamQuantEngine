#pragma once
// TokenFrequencyTable.h — N-gram frequency table for token stream analysis.

#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llmquant {

/// A sequence of N tokens forming an N-gram.
struct NGram {
    std::vector<std::string> tokens;
    std::size_t              n{0};

    bool operator==(const NGram& other) const noexcept {
        return tokens == other.tokens;
    }
    bool operator<(const NGram& other) const noexcept {
        return tokens < other.tokens;
    }
};

/// Tracks unigram and N-gram frequencies for a token stream.
///
/// Supports:
/// - Unigram counts with O(1) lookup
/// - N-gram counts using a sorted map keyed by token vector
/// - Conditional probability P(token | context)
/// - Top-K most frequent unigrams
/// - Feed sequences to extract all N-grams up to a given N
class TokenFrequencyTable {
public:
    /// Construct an empty frequency table.
    TokenFrequencyTable() = default;

    // ------------------------------------------------------------------
    // Ingestion
    // ------------------------------------------------------------------

    /// Increment the unigram count for a single token.
    void add_token(const std::string& token);

    /// Increment the count for an explicit N-gram.
    void add_ngram(const std::vector<std::string>& tokens, std::size_t n);

    /// Process a sequence of tokens, extracting all N-grams of length 1..max_n.
    void feed_sequence(const std::vector<std::string>& tokens, std::size_t max_n = 3);

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Return the unigram count for a token (0 if unseen).
    std::size_t frequency(const std::string& token) const;

    /// Return the N-gram count for the given token vector (0 if unseen).
    std::size_t ngram_frequency(const std::vector<std::string>& ngram) const;

    /// Return the top-K unigrams by count, sorted descending.
    std::vector<std::pair<std::string, std::size_t>> top_k(std::size_t k) const;

    /// Compute P(token | context) = count(context + token) / count(context).
    /// Returns 0.0 if the context has never been seen.
    double conditional_probability(
        const std::string&              token,
        const std::vector<std::string>& context) const;

    /// Total number of tokens ingested via add_token / feed_sequence.
    std::size_t total_tokens() const noexcept;

    /// Number of distinct unigrams seen.
    std::size_t unique_count() const noexcept;

    /// Reset all counters.
    void reset();

private:
    std::unordered_map<std::string, std::size_t> unigram_counts_;
    std::map<std::vector<std::string>, std::size_t> ngram_counts_;
    std::size_t total_tokens_{0};
};

}  // namespace llmquant
