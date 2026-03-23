// TokenFrequencyTable.cpp — N-gram frequency table implementation.

#include "TokenFrequencyTable.h"

#include <algorithm>
#include <cassert>
#include <vector>

namespace llmquant {

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------

void TokenFrequencyTable::add_token(const std::string& token) {
    unigram_counts_[token]++;
    total_tokens_++;
}

void TokenFrequencyTable::add_ngram(
    const std::vector<std::string>& tokens, std::size_t /*n*/)
{
    if (tokens.empty()) return;
    ngram_counts_[tokens]++;
}

void TokenFrequencyTable::feed_sequence(
    const std::vector<std::string>& tokens, std::size_t max_n)
{
    if (tokens.empty()) return;
    const std::size_t len = tokens.size();

    for (std::size_t i = 0; i < len; ++i) {
        for (std::size_t n = 1; n <= max_n && (i + n) <= len; ++n) {
            std::vector<std::string> gram(tokens.begin() + static_cast<std::ptrdiff_t>(i),
                                          tokens.begin() + static_cast<std::ptrdiff_t>(i + n));
            if (n == 1) {
                unigram_counts_[gram[0]]++;
                total_tokens_++;
            } else {
                ngram_counts_[gram]++;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

std::size_t TokenFrequencyTable::frequency(const std::string& token) const {
    auto it = unigram_counts_.find(token);
    return it != unigram_counts_.end() ? it->second : 0;
}

std::size_t TokenFrequencyTable::ngram_frequency(
    const std::vector<std::string>& ngram) const
{
    auto it = ngram_counts_.find(ngram);
    return it != ngram_counts_.end() ? it->second : 0;
}

std::vector<std::pair<std::string, std::size_t>>
TokenFrequencyTable::top_k(std::size_t k) const {
    std::vector<std::pair<std::string, std::size_t>> pairs(
        unigram_counts_.begin(), unigram_counts_.end());

    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    if (k < pairs.size()) {
        pairs.resize(k);
    }
    return pairs;
}

double TokenFrequencyTable::conditional_probability(
    const std::string&              token,
    const std::vector<std::string>& context) const
{
    if (context.empty()) {
        // P(token) = unigram frequency
        if (total_tokens_ == 0) return 0.0;
        return static_cast<double>(frequency(token)) /
               static_cast<double>(total_tokens_);
    }

    // Context count: look up the context N-gram
    std::size_t ctx_count = ngram_frequency(context);
    if (ctx_count == 0) return 0.0;

    // Numerator: count(context ++ [token])
    std::vector<std::string> full_ngram = context;
    full_ngram.push_back(token);
    std::size_t full_count = ngram_frequency(full_ngram);

    return static_cast<double>(full_count) / static_cast<double>(ctx_count);
}

std::size_t TokenFrequencyTable::total_tokens() const noexcept {
    return total_tokens_;
}

std::size_t TokenFrequencyTable::unique_count() const noexcept {
    return unigram_counts_.size();
}

void TokenFrequencyTable::reset() {
    unigram_counts_.clear();
    ngram_counts_.clear();
    total_tokens_ = 0;
}

}  // namespace llmquant
