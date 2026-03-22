#pragma once

/// @file SignalCausalChainTracer.h
/// @brief Traces the top-N token contributions that caused a trade signal.
///
/// ## Motivation
/// When a trade signal fires, operators need to understand *why* — which tokens
/// in the recent stream drove the bias/volatility move.  This module keeps a
/// rolling window of (token, contribution) pairs and, on demand, returns the
/// top-N contributors ranked by absolute contribution magnitude.
///
/// ## Algorithm
/// - `record_token(token, contribution)` appends to a ring buffer of size
///   `window`.  Contribution is typically the bias delta caused by that token.
/// - `trace(n)` returns up to N entries from the current window sorted
///   descending by |contribution| — the "causal chain" for the most recent signal.
/// - `on_strong_token` fires whenever |contribution| exceeds `strong_threshold`,
///   immediately (outside lock), giving a real-time alert on high-impact tokens.
///
/// ## Feature flag
/// `LLMQUANT_ENABLE_CAUSAL_TRACER` (default ON).
///
/// Thread-safe: all public methods are thread-safe.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace llmquant {

class SignalCausalChainTracer {
public:
    struct TokenContribution {
        std::string token;
        double      contribution{0.0};
    };

    struct Config {
        /// Rolling window capacity (number of recent tokens tracked). Default 128.
        int window{128};

        /// Absolute contribution magnitude above which `on_strong_token` fires. Default 0.05.
        double strong_threshold{0.05};

        /// Fired immediately when a token's |contribution| exceeds strong_threshold.
        /// Parameters: (token, contribution).
        std::function<void(const std::string&, double)> on_strong_token;
    };

    explicit SignalCausalChainTracer(Config cfg = Config{});

    // -----------------------------------------------------------------------
    // Core
    // -----------------------------------------------------------------------

    /// Record one token and its bias contribution to the rolling window.
    void record_token(const std::string& token, double contribution);

    /// Return the top-N tokens by |contribution| from the current window.
    /// If fewer than N tokens are in the window, all available are returned.
    [[nodiscard]] std::vector<TokenContribution> trace(int n = 5) const;

    /// Return the full window (temporal order, oldest first).
    [[nodiscard]] std::vector<TokenContribution> window_snapshot() const;

    // -----------------------------------------------------------------------
    // Accessors (lock-free)
    // -----------------------------------------------------------------------

    /// Total tokens recorded since construction or last reset().
    [[nodiscard]] uint64_t total_tokens() const noexcept {
        return total_.load(std::memory_order_relaxed);
    }

    /// Number of strong-token events fired.
    [[nodiscard]] uint64_t strong_token_events() const noexcept {
        return strong_events_.load(std::memory_order_relaxed);
    }

    /// Absolute contribution of the most impactful token in the current window.
    [[nodiscard]] double max_contribution() const noexcept {
        return max_contrib_.load(std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    void reset();
    void update_config(const Config& cfg);

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    [[nodiscard]] std::string to_stats_json() const;

private:
    mutable std::mutex mutex_;
    Config cfg_;

    // Ring buffer
    std::vector<TokenContribution> buf_;
    int head_{0};
    int fill_{0};

    // Atomic outputs
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> strong_events_{0};
    std::atomic<double>   max_contrib_{0.0};
};

} // namespace llmquant
