#pragma once

/// @file TokenSaliencyRanker.h
/// @brief Ranks token types by their marginal saliency — the signed gradient
///        of cumulative bias with respect to token frequency.
///
/// ## Motivation
/// Not all tokens are equal.  A token that consistently precedes large positive
/// bias shifts is a high-saliency bullish driver; one that precedes small random
/// shifts has near-zero saliency.  By maintaining per-token statistics on
/// (frequency, cumulative bias contribution, contribution variance) we produce:
///   - Saliency score s_i = mean_contribution_i / (std_i + ε)  (signal-to-noise)
///   - Top-K bullish and bearish saliency tokens.
///
/// The token vocabulary is modelled by hashed IDs modulo `max_tokens` (default
/// 512) so memory is bounded.  Unlike TokenInfluenceAttributor which tracks
/// per-token cumulative weight across all time, this module normalises by
/// frequency to produce a per-occurrence saliency.
///
/// Fires `on_top_token_change(token_id, saliency)` when the highest-saliency
/// token changes.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_SALIENCY_RANKER` (default ON).
///
/// Thread-safe.

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class TokenSaliencyRanker {
public:
    static constexpr int kMaxTokens = 512;
    static constexpr int kTopK      = 5;

    struct Config {
        /// EMA decay for per-token mean/var update. Default 0.10.
        double ema_alpha{0.10};

        /// Minimum per-token observations before including in ranking. Default 3.
        int min_token_count{3};

        /// Fired when top-saliency token changes (outside lock).
        std::function<void(int token_id, double saliency)> on_top_token_change;
    };

    explicit TokenSaliencyRanker(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record token_id (hashed, modulo kMaxTokens) with its bias contribution.
    void record(int token_id, double bias_contribution);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Saliency score for a token ID.
    [[nodiscard]] double saliency(int token_id) const;

    /// Top-K bullish token IDs sorted by descending saliency score.
    [[nodiscard]] std::array<int, kTopK> top_bullish() const;

    /// Top-K bearish token IDs sorted by ascending (most negative) saliency.
    [[nodiscard]] std::array<int, kTopK> top_bearish() const;

    /// Total observations recorded.
    [[nodiscard]] uint64_t total_records() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    mutable std::mutex mutex_;
    Config cfg_;

    struct TokenStats {
        double  mean{0.0};   // EMA of contribution
        double  var{0.001};  // EMA of squared deviation (initialised to small positive)
        int     count{0};
    };
    std::vector<TokenStats> stats_;    // [kMaxTokens]
    int prev_top_{-1};

    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
