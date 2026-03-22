#pragma once

/// @file TokenNgramProfiler.h
/// @brief Sliding-window n-gram frequency tracker for repeated narrative patterns.
///
/// Maintains frequency counts for all 2-grams and 3-grams seen in the token
/// stream.  Fires a callback when any n-gram exceeds a configurable frequency
/// threshold — a signal that the same phrase is repeating, which may indicate
/// an adversarial prompt injection, a stuck LLM, or genuine narrative momentum.
///
/// Internally stores a fixed-size token history and updates the frequency map
/// on each push.  The map is bounded by max_ngrams to prevent unbounded growth.
///
/// Thread-safe: all public methods are thread-safe.
///
/// Feature flag: LLMQUANT_ENABLE_TOKEN_NGRAM_PROFILER (default ON).

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

class TokenNgramProfiler {
public:
    struct NgramEntry {
        std::string ngram;       ///< Space-joined n-gram tokens.
        uint64_t    count{0};    ///< Frequency since last reset.
        int         n{0};        ///< Gram size (2 or 3).
    };

    struct Config {
        /// Track 2-grams.  Default true.
        bool enable_bigrams{true};

        /// Track 3-grams.  Default true.
        bool enable_trigrams{true};

        /// Max distinct n-grams tracked before LRU eviction.  Default 8192.
        uint32_t max_ngrams{8192};

        /// Frequency count above which on_hot_ngram fires.  Default 10.
        uint32_t hot_threshold{10};

        /// Fired when an n-gram crosses hot_threshold.
        /// Parameters: (ngram_string, count, n).
        std::function<void(const std::string&, uint64_t, int)> on_hot_ngram;
    };

    explicit TokenNgramProfiler(Config cfg = Config{});

    // Non-copyable.
    TokenNgramProfiler(const TokenNgramProfiler&)            = delete;
    TokenNgramProfiler& operator=(const TokenNgramProfiler&) = delete;

    /// Record one token from the stream.
    void push(const std::string& token);

    /// Top-N entries by frequency (bigrams and trigrams combined).
    [[nodiscard]] std::vector<NgramEntry> top_by_frequency(uint32_t n) const;

    /// Frequency of an exact n-gram string (space-joined).
    [[nodiscard]] uint64_t frequency(const std::string& ngram) const;

    /// Total tokens pushed.
    [[nodiscard]] uint64_t total_tokens() const noexcept {
        return total_tokens_.load(std::memory_order_relaxed);
    }

    /// Total distinct n-grams tracked.
    [[nodiscard]] uint32_t distinct_ngrams() const;

    /// Total hot-ngram events fired.
    [[nodiscard]] uint64_t hot_events() const noexcept {
        return hot_events_.load(std::memory_order_relaxed);
    }

    /// Reset frequency counts and token history.
    void reset();

    /// Update config and reset.
    void update_config(const Config& cfg);

    /// JSON diagnostics: top-5 n-grams + counters.
    [[nodiscard]] std::string to_stats_json() const;

private:
    Config cfg_;

    std::deque<std::string>                    history_; // recent tokens
    std::unordered_map<std::string, uint64_t>  freq_;    // ngram → count
    std::vector<std::string>                   insert_order_; // for LRU eviction

    std::atomic<uint64_t> total_tokens_{0};
    std::atomic<uint64_t> hot_events_{0};

    mutable std::mutex mutex_;

    void record_ngram(const std::string& key, int n);
};

} // namespace llmquant
