#pragma once
// TokenProfiler.h — Profile token stream performance statistics.

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

// ---------------------------------------------------------------------------
// TokenProfile: per-token statistics
// ---------------------------------------------------------------------------

struct TokenProfile {
    std::string token;
    std::size_t frequency{0};
    double avg_position{0.0};   ///< average index position in the stream
    std::size_t first_seen_idx{0};
    std::size_t last_seen_idx{0};
};

// ---------------------------------------------------------------------------
// StreamProfile: aggregate stream statistics
// ---------------------------------------------------------------------------

struct StreamProfile {
    std::size_t total_tokens{0};
    std::size_t unique_tokens{0};
    std::size_t vocab_size{0};
    double entropy{0.0};                    ///< Shannon entropy of token distribution
    std::vector<TokenProfile> top_k_tokens; ///< sorted by frequency descending
};

// ---------------------------------------------------------------------------
// TokenProfiler
// ---------------------------------------------------------------------------

/// Profiles a stream of tokens, computing frequency, position, and entropy.
class TokenProfiler {
public:
    TokenProfiler() = default;
    ~TokenProfiler() = default;

    // Non-copyable (holds mutable accumulation state)
    TokenProfiler(const TokenProfiler&) = delete;
    TokenProfiler& operator=(const TokenProfiler&) = delete;

    // Movable
    TokenProfiler(TokenProfiler&&) = default;
    TokenProfiler& operator=(TokenProfiler&&) = default;

    /// Feed a single token into the profiler.
    void feed(const std::string& token);

    /// Feed a batch of tokens.
    void feed_batch(const std::vector<std::string>& tokens);

    /// Compute and return the full stream profile.
    /// @param top_k  Number of most-frequent tokens to include in StreamProfile.
    StreamProfile profile(std::size_t top_k = 10) const;

    /// Compute Shannon entropy of the current token distribution.
    double entropy() const;

    /// Reset all accumulated statistics.
    void reset();

    /// Return total number of tokens seen so far.
    std::size_t total_tokens() const noexcept { return total_tokens_; }

    /// Return number of unique tokens seen.
    std::size_t unique_count() const noexcept { return stats_.size(); }

private:
    struct TokenStats {
        std::size_t frequency{0};
        double position_sum{0.0};
        std::size_t first_seen{0};
        std::size_t last_seen{0};
    };

    std::unordered_map<std::string, TokenStats> stats_;
    std::size_t total_tokens_{0};
};

} // namespace llmquant
