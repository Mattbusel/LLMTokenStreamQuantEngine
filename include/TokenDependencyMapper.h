#pragma once

/// @file TokenDependencyMapper.h
/// @brief Tracks pairwise token co-occurrence within a sliding window to reveal
///        compound narrative structures in the LLM token stream.
///
/// ## Motivation
/// Individual token weights capture sentiment in isolation.  Compound narratives
/// (e.g., "Fed" + "rate-hike" + "inflation" co-occurring in the same 20-token
/// span) carry qualitatively different signal from any single token.  This module
/// maintains a co-occurrence count matrix and fires `on_cluster_detected` when
/// a group of tokens mutually co-occur above a density threshold — indicating an
/// emergent narrative cluster.
///
/// ## Algorithm
/// - Keeps a sliding window of the last `window` token IDs (integer keys).
/// - On each `record_token(id)`, increments co-occurrence counts for every
///   (id, x) pair where x is already in the current window.
/// - Periodically (every `recompute_interval` inserts), scans co-occurrence
///   counts and reports clusters: sets of ≥ `min_cluster_size` token IDs whose
///   pairwise count all exceed `min_count`.
/// - Fires `on_cluster_detected(cluster_ids)` outside the lock.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_DEPENDENCY_MAPPER` (default ON).
///
/// Thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmquant {

class TokenDependencyMapper {
public:
    struct Config {
        /// Sliding window of token IDs tracked for co-occurrence. Default 32.
        int window{32};

        /// Minimum pairwise co-occurrence count to consider an edge. Default 3.
        int min_count{3};

        /// Minimum number of tokens in a reported cluster. Default 3.
        int min_cluster_size{3};

        /// Recompute cluster search every N inserts. Default 16.
        int recompute_interval{16};

        /// Fired when a cluster is detected (outside lock).
        /// Parameter: sorted vector of token IDs in the cluster.
        std::function<void(const std::vector<int>&)> on_cluster_detected;
    };

    explicit TokenDependencyMapper(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Data intake
    // -----------------------------------------------------------------------

    /// Record a token by integer ID. Updates co-occurrence matrix.
    void record_token(int token_id);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Co-occurrence count for the given token pair (symmetric).
    [[nodiscard]] int cooccurrence(int id_a, int id_b) const;

    /// Total tokens recorded since construction or last reset().
    [[nodiscard]] uint64_t total_tokens() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Number of cluster-detected events fired.
    [[nodiscard]] uint64_t cluster_events() const noexcept {
        return cluster_events_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_stats_json() const;

    void reset();
    void update_config(const Config& cfg);

private:
    void find_clusters_locked(std::vector<std::vector<int>>& out) const;

    mutable std::mutex mutex_;
    Config cfg_;

    // Circular window of recent token IDs
    std::vector<int> win_buf_;
    int win_head_{0};
    int win_fill_{0};

    // Co-occurrence counts: key = min(a,b)*100003 + max(a,b) (collision-safe for typical IDs)
    std::unordered_map<uint64_t, int> cooc_;

    int insert_count_{0};   // triggers recompute every recompute_interval

    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> cluster_events_{0};
};

} // namespace llmquant
