#pragma once

/// @file TokenSentimentGraphBuilder.h
/// @brief Builds a weighted directed graph of token-to-token sentiment
///        influence from co-occurrence and conditional bias transitions.
///
/// ## Motivation
/// Individual tokens carry sentiment weight, but their effects are not
/// independent — "crash" amplifies the sentiment of adjacent "market" more
/// than standalone.  By building a directed graph where edge weight
/// w(a→b) = E[bias_b | a appears in previous N tokens] − E[bias_b],
/// we quantify token influence relationships and detect:
///   - High in-degree nodes (amplifiers / sentiment hubs).
///   - High out-degree nodes (sentiment seeds / initiators).
///   - Strongly connected components (self-reinforcing sentiment clusters).
///
/// ## Algorithm
/// - Maintains a sliding window of the last `window` (token_id, bias) pairs.
/// - On each new (token_id, bias) pair, updates:
///     edge_sum[prev_token_id][token_id] += bias  (conditional accumulator)
///     edge_count[prev_token_id][token_id] += 1
/// - Edge weight w(a→b) = edge_sum[a][b] / edge_count[a][b].
/// - Fires `on_hub_detected(token_id, in_degree_weight)` when in-degree
///   weight of any token exceeds `hub_threshold`.
/// - Tracks top-K (K=3) highest in-degree-weight token IDs.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_SENTIMENT_GRAPH` (default ON).
///
/// Thread-safe.
///
/// @note Token IDs are capped at max_tokens (default 256) for bounded memory.

#include <atomic>
#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class TokenSentimentGraphBuilder {
public:
    static constexpr int kMaxTokens = 256;
    static constexpr int kTopK      = 3;

    struct Config {
        /// Sliding window of (token, bias) pairs for recency weighting. Default 200.
        int window{200};

        /// In-degree weight above this fires on_hub_detected. Default 0.5.
        double hub_threshold{0.5};

        /// Minimum edge count before reporting an edge weight. Default 3.
        int min_edge_count{3};

        /// Fired when a token's total in-degree weight exceeds hub_threshold (outside lock).
        std::function<void(int token_id, double in_weight)> on_hub_detected;
    };

    explicit TokenSentimentGraphBuilder(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record token_id ∈ [0, kMaxTokens) with its current bias contribution.
    void record(int token_id, double bias);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Edge weight w(from→to) = conditional mean bias contribution.
    [[nodiscard]] double edge_weight(int from, int to) const;

    /// Sum of incoming edge weights for a token (in-degree weight).
    [[nodiscard]] double in_degree_weight(int token_id) const;

    /// Top-K token IDs by in-degree weight (descending).
    [[nodiscard]] std::array<int, kTopK> top_hubs() const;

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

    // Flat edge tables (kMaxTokens × kMaxTokens)
    std::vector<double> edge_sum_;    // edge_sum[from*kMaxTokens + to]
    std::vector<int>    edge_count_;  // edge_count[from*kMaxTokens + to]

    // Ring buffer of (token_id, bias) for windowed eviction
    struct Entry { int token_id; double bias; };
    std::vector<Entry> ring_;
    int ring_head_{0};
    int ring_fill_{0};

    int prev_token_{-1};

    std::atomic<uint64_t> total_{0};
};

} // namespace llmquant
